# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Standalone version of the Llama cookbook finetuning script
# Adapted from https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/finetuning.py

import dataclasses
import os
import random
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

import fire
import numpy as np
import torch
import torch.optim as optim
from accelerate.utils import is_xpu_available

from peft import get_peft_model, PeftModel, LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import Dataset, load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llama_cookbook.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Configuration classes
@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"  # Model name or path
    tokenizer_name: Optional[str] = None  # Tokenizer name or path
    enable_fsdp: bool = False  # Enable FSDP (Fully Sharded Data Parallel)
    low_cpu_fsdp: bool = False  # Enable low CPU usage FSDP
    run_validation: bool = True  # Run validation during training
    batch_size_training: int = 1  # Batch size for training
    gradient_accumulation_steps: int = 1  # Number of gradient accumulation steps
    num_epochs: int = 3  # Number of training epochs
    num_workers_dataloader: int = 1  # Number of workers for dataloader
    lr: float = 1e-4  # Learning rate
    weight_decay: float = 0.0  # Weight decay
    gamma: float = 0.85  # Gamma for lr scheduler
    seed: int = 42  # Random seed
    use_fp16: bool = False  # Use FP16 precision
    mixed_precision: bool = True  # Use mixed precision
    val_batch_size: int = 1  # Batch size for validation
    context_length: int = 4096  # Maximum context length
    batching_strategy: str = "packing"  # Batching strategy: 'packing' or 'padding'
    dataset_path: str = "formatted_data"  # Path to dataset
    use_peft: bool = True  # Use PEFT (Parameter-Efficient Fine-Tuning)
    peft_method: str = "lora"  # PEFT method: 'lora', 'prefix', etc.
    quantization: Union[bool, str] = False  # Quantization: False, '4bit', '8bit'
    use_fast_kernels: bool = True  # Use Flash Attention
    freeze_layers: bool = False  # Freeze layers of the model
    freeze_LLM_only: bool = False  # Freeze only the LLM part
    num_freeze_layers: int = 0  # Number of layers to freeze
    from_peft_checkpoint: Optional[str] = None  # Path to PEFT checkpoint
    use_wandb: bool = False  # Use Weights & Biases for logging
    output_dir: str = "output"  # Output directory
    save_strategy: str = "epoch"  # Save strategy: 'epoch', 'steps'
    save_steps: int = 500  # Save steps
    eval_steps: int = 500  # Evaluation steps
    logging_steps: int = 100  # Logging steps
    save_total_limit: int = 3  # Maximum number of checkpoints to save
    push_to_hub: bool = False  # Push model to Hugging Face Hub
    hub_model_id: Optional[str] = None  # Model ID for Hugging Face Hub
    hub_token: Optional[str] = None  # Token for Hugging Face Hub

@dataclass
class FSDPConfig:
    mixed_precision: bool = True  # Use mixed precision
    use_fp16: bool = False  # Use FP16 precision
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  # FSDP sharding strategy
    fsdp_activation_checkpointing: bool = True  # Enable activation checkpointing for FSDP
    pure_bf16: bool = False  # Use pure BF16 precision
    optimizer: str = "adamw"  # Optimizer: 'adamw', 'anyprecision'
    fsdp_cpu_offload: bool = False  # Enable CPU offloading for FSDP
    hsdp: bool = False  # Enable HSDP (Hybrid Sharded Data Parallel)
    replica_group_size: int = 1  # Replica group size for HSDP
    sharding_group_size: int = 1  # Sharding group size for HSDP

@dataclass
class QuantizationConfig:
    compute_dtype: torch.dtype = torch.float16  # Computation data type
    quant_type: str = "nf4"  # Quantization type: 'nf4', 'fp4'
    double_quant: bool = True  # Use double quantization
    
    def create_bnb_config(self, quantization: str) -> BitsAndBytesConfig:
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_quant_type=self.quant_type,
                bnb_4bit_use_double_quant=self.double_quant,
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"Quantization {quantization} not supported. Use '4bit' or '8bit'.")

@dataclass
class WandbConfig:
    project: str = "llama-finetuning"  # Weights & Biases project name
    entity: Optional[str] = None  # Weights & Biases entity name
    group: Optional[str] = None  # Weights & Biases group name
    name: Optional[str] = None  # Weights & Biases run name
    tags: List[str] = field(default_factory=list)  # Weights & Biases tags

# Utility functions
def update_config(config_objects, **kwargs):
    """Update configuration objects with keyword arguments."""
    if not isinstance(config_objects, tuple):
        config_objects = (config_objects,)
    
    for config in config_objects:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

def setup_wandb(train_config, fsdp_config, **kwargs):
    """Setup Weights & Biases for logging."""
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    
    wandb_config = WandbConfig()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(dataclasses.asdict(train_config))
    run.config.update(dataclasses.asdict(fsdp_config), allow_val_change=True)
    return run


# Dataset handling functions
def load_dataset_from_path(dataset_path, split='train'):
    """Load dataset from path."""
    logger.info(f"Loading {split} dataset from {dataset_path}")
    
    # Check if dataset_path is a directory or a file
    if os.path.isdir(dataset_path):
        # If it's a directory, look for JSON or CSV files
        files = [f for f in os.listdir(dataset_path) if f.endswith('.json') or f.endswith('.csv')]
        if not files:
            raise ValueError(f"No JSON or CSV files found in {dataset_path}")
            
        # Filter files based on split
        split_files = [f for f in files if split in f.lower()]
        if not split_files and split != 'train':
            logger.warning(f"No files found for split '{split}', using all files")
            split_files = files
            
        file_path = os.path.join(dataset_path, split_files[0])
        if file_path.endswith('.json'):
            return Dataset.from_json(file_path)
        elif file_path.endswith('.csv'):
            return Dataset.from_csv(file_path)
    else:
        # If it's a file, load it directly
        if dataset_path.endswith('.json'):
            return Dataset.from_json(dataset_path)
        elif dataset_path.endswith('.csv'):
            return Dataset.from_csv(dataset_path)
        else:
            # Try to load as a Hugging Face dataset
            try:
                return load_dataset(dataset_path, split=split)
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise

def preprocess_dataset(dataset, tokenizer, max_length=4096):
    """Preprocess dataset for training."""
    # Check if dataset has the expected columns
    if 'text' not in dataset.column_names:
        # Try to find a suitable text column
        text_columns = [col for col in dataset.column_names if 'text' in col.lower()]
        if text_columns:
            logger.info(f"Using '{text_columns[0]}' as text column")
            dataset = dataset.rename_column(text_columns[0], 'text')
        else:
            # If no text column is found, try to use the first column
            logger.warning(f"No text column found, using '{dataset.column_names[0]}' as text column")
            dataset = dataset.rename_column(dataset.column_names[0], 'text')
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != 'text'],
    )
    
    return tokenized_dataset

# FSDP utility functions
def setup_fsdp():
    """Setup for Fully Sharded Data Parallel training."""
    torch.distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo"
    )

def clear_gpu_cache(rank=None):
    """Clear GPU cache."""
    if rank == 0 or rank is None:
        if is_xpu_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

# Model utility functions
def print_model_size(model, config, rank=0):
    """Print model size information."""
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params / 1e6:.2f}M")

def generate_peft_config(train_config, kwargs):
    """Generate PEFT configuration."""
    if train_config.peft_method.lower() == "lora":
        # Default LoRA configuration
        lora_config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        
        # Update with any provided kwargs
        for key in ["lora_r", "lora_alpha", "lora_dropout"]:
            if key in kwargs:
                # Convert from lora_r to r, etc.
                config_key = key.replace("lora_", "")
                lora_config[config_key] = kwargs[key]
        
        return LoraConfig(**lora_config)
    else:
        raise ValueError(f"PEFT method {train_config.peft_method} not supported")

def main(**kwargs):
    """Main function for fine-tuning Llama models."""
    # Create and update configuration
    train_config = TrainConfig()
    fsdp_config = FSDPConfig()
    update_config((train_config, fsdp_config), **kwargs)
    
    # Set random seeds for reproducibility
    set_seed(train_config.seed)
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    
    # Setup for distributed training if enabled
    local_rank = 0
    rank = 0
    world_size = 1
    
    if train_config.enable_fsdp:
        setup_fsdp()
        # Get distributed training information
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Setup device
        if torch.distributed.is_initialized():
            if is_xpu_available():
                torch.xpu.set_device(local_rank)
            elif torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            clear_gpu_cache(rank)
    
    # Setup wandb logging if enabled
    wandb_run = None
    if train_config.use_wandb and (not train_config.enable_fsdp or rank == 0):
        wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)
    
    # Setup quantization if enabled
    bnb_config = None
    if train_config.quantization:
        # Handle boolean quantization flag (default to 8-bit)
        if isinstance(train_config.quantization, bool) and train_config.quantization:
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. "
                "Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            train_config.quantization = "8bit"
        
        # Check compatibility with FSDP
        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError(
                "8bit quantization is not supported with FSDP, please use 4bit quantization"
            )
        
        # Create quantization config
        quant_config = QuantizationConfig()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # Load the pre-trained model and tokenizer
    logger.info(f"Loading model: {train_config.model_name}")
    
    # Determine if we should use cache (disable for FSDP)
    use_cache = False if train_config.enable_fsdp else None
    
    # Load model configuration
    config = AutoConfig.from_pretrained(train_config.model_name)
    model_type = getattr(config, 'model_type', '').lower()
    
    # Load the model based on its type
    if model_type == 'llama':
        logger.info("Loading LlamaForCausalLM model")
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else "auto",
        )
    else:
        logger.warning(f"Model type '{model_type}' not explicitly supported, attempting to load as a causal LM")
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else "auto",
        )
    
    # Load the tokenizer
    tokenizer_name = train_config.tokenizer_name or train_config.model_name
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure the tokenizer has a padding token
    if not tokenizer.pad_token_id:
        logger.info("Setting pad_token_id to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Check for tokenizer and model embedding size mismatch
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.warning("Resizing the embedding matrix to match the tokenizer vocab size")
        model.resize_token_embeddings(len(tokenizer))
    
    # Print model size information
    print_model_size(model, train_config, rank)
    
    # Apply PEFT if enabled
    if train_config.use_peft:
        logger.info(f"Applying PEFT using method: {train_config.peft_method}")
        
        if train_config.from_peft_checkpoint:
            # Load from existing PEFT checkpoint
            logger.info(f"Loading PEFT model from checkpoint: {train_config.from_peft_checkpoint}")
            model = PeftModel.from_pretrained(
                model, train_config.from_peft_checkpoint, is_trainable=True
            )
        else:
            # Create new PEFT model
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
            
            # Log trainable parameters
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
            else:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                all_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/all_params:.2%} of all parameters)")
    
    # Move model to appropriate device if not using FSDP or quantization
    if not train_config.enable_fsdp and not train_config.quantization:
        if is_xpu_available():
            model.to("xpu:0")
            logger.info("Model moved to XPU")
        elif torch.cuda.is_available():
            model.to("cuda")
            logger.info("Model moved to CUDA")
    
    # Load and preprocess the dataset
    logger.info(f"Loading dataset from: {train_config.dataset_path}")
    
    # Load training dataset
    try:
        train_dataset = load_dataset_from_path(train_config.dataset_path, split='train')
        logger.info(f"Training dataset loaded: {len(train_dataset)} examples")
    except Exception as e:
        logger.error(f"Error loading training dataset: {e}")
        raise
    
    # Preprocess training dataset
    train_dataset = preprocess_dataset(
        train_dataset, 
        tokenizer, 
        max_length=train_config.context_length
    )
    
    # Load validation dataset if validation is enabled
    eval_dataset = None
    if train_config.run_validation:
        try:
            eval_dataset = load_dataset_from_path(train_config.dataset_path, split='validation')
            logger.info(f"Validation dataset loaded: {len(eval_dataset)} examples")
            
            # Preprocess validation dataset
            eval_dataset = preprocess_dataset(
                eval_dataset, 
                tokenizer, 
                max_length=train_config.context_length
            )
        except Exception as e:
            logger.warning(f"Could not load validation dataset: {e}. Trying 'test' split instead.")
            try:
                eval_dataset = load_dataset_from_path(train_config.dataset_path, split='test')
                logger.info(f"Test dataset loaded: {len(eval_dataset)} examples")
                
                # Preprocess test dataset
                eval_dataset = preprocess_dataset(
                    eval_dataset, 
                    tokenizer, 
                    max_length=train_config.context_length
                )
            except Exception as e2:
                logger.warning(f"Could not load test dataset either: {e2}. Proceeding without validation.")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size_training,
        per_device_eval_batch_size=train_config.val_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.lr,
        weight_decay=train_config.weight_decay,
        warmup_ratio=0.03,
        bf16=torch.cuda.is_bf16_supported() and not train_config.use_fp16,
        fp16=train_config.use_fp16,
        evaluation_strategy="epoch" if train_config.run_validation and eval_dataset else "no",
        save_strategy=train_config.save_strategy,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        logging_steps=train_config.logging_steps,
        save_total_limit=train_config.save_total_limit,
        load_best_model_at_end=train_config.run_validation and eval_dataset is not None,
        report_to="wandb" if train_config.use_wandb else "none",
        push_to_hub=train_config.push_to_hub,
        hub_model_id=train_config.hub_model_id,
        hub_token=train_config.hub_token,
    )
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    if not train_config.enable_fsdp or rank == 0:
        logger.info(f"Saving model to {train_config.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(train_config.output_dir)
        
        # Log results if using wandb
        if train_config.use_wandb and wandb_run:
            wandb_run.finish()
    
    return {"status": "success", "output_dir": train_config.output_dir}


if __name__ == "__main__":
    # Example usage
    fire.Fire(main)


if __name__ == "__main__":
    fire.Fire(main)