#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Llama models on Yanomami language data.
This script uses the installed llama-cookbook package.

Usage:
    python yanomami_finetune.py --phase=1 --model_name="meta-llama/Meta-Llama-3.1-8B" --tokenizer_path="./fixed_tokenizer"
"""

import argparse
import os
import logging
import sys
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Dict, Any, Union, Callable

import torch
from transformers import set_seed

# We'll import specific modules from the installed llama-cookbook package when needed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"finetune_yanomami.log")
    ]
)
logger = logging.getLogger(__name__)

# Define configuration classes here to avoid import issues
@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer_name: Optional[str] = None
    dataset_path: str = "./"
    phase: int = 1
    output_dir: str = "./output"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    context_length: int = 4096
    use_peft: bool = False
    lora_r: int = 128
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
    ])
    use_fp16: bool = False
    enable_fsdp: bool = False
    run_validation: bool = True
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: int = 100
    seed: int = 42
    data_seed: int = 42

    # Hardcoded file paths
    phase1_file: str = "phase1_data.txt"
    phase2_file: str = "phase2_data.txt"

@dataclass
class FSDPConfig:
    pure_bf16: bool = False
    optimizer: str = "adamw"
    fsdp_activation_checkpointing: bool = True

@dataclass
class QuantizationConfig:
    bits: int = 4
    group_size: int = 128
    double_quant: bool = True

@dataclass
class WandbConfig:
    project: str = "llama-yanomami"
    run_name: str = "yanomami-finetune"
    watch_model: bool = False
    save_model: bool = False

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama models on Yanomami language data")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="Training phase: 1 for translation-based training, 2 for bilingual next token prediction"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Name or path of the model to fine-tune"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./fixed_tokenizer",
        help="Path to the extended tokenizer"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./formatted_data",
        help="Path to the formatted training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Micro batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--cutoff_len",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=1000,
        help="Validation set size"
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Whether to use PEFT for fine-tuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=128,
        help="LoRA rank parameter"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout parameter"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llama-yanomami",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="W&B run name"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Whether to use FP16 precision"
    )
    parser.add_argument(
        "--enable_fsdp",
        action="store_true",
        help="Whether to enable FSDP for distributed training"
    )
    return parser.parse_args()

def preprocess_dataset(dataset, tokenizer, max_length=4096, is_train=True):
    """
    Preprocess the dataset for training, ensuring proper padding and truncation.
    This addresses the data formatting issue mentioned in the error message.
    """
    logger.info(f"Preprocessing dataset with {len(dataset)} examples")
    
    def tokenize_function(examples):
        # Handle different data formats based on the dataset structure
        if "text" in examples:
            # Standard text format
            texts = examples["text"]
        elif "input" in examples and "output" in examples:
            # Input-output format (common in instruction tuning)
            inputs = examples["input"]
            outputs = examples["output"]
            if isinstance(inputs, list) and isinstance(outputs, list):
                texts = [f"{i}\n\n{o}" for i, o in zip(inputs, outputs)]
            else:
                texts = f"{inputs}\n\n{outputs}"
        else:
            # Try to find any text-like field
            text_fields = [k for k in examples.keys() if any(t in k.lower() for t in ["text", "content", "data"])]
            if text_fields:
                texts = examples[text_fields[0]]
            else:
                # Fallback: convert the entire example to a string
                texts = str(examples)
        
        # Ensure texts is properly formatted
        if isinstance(texts, list):
            # Handle case where text is a list of strings
            texts = [t if isinstance(t, str) else str(t) for t in texts]
        elif not isinstance(texts, str):
            # Convert to string if not already a string
            texts = str(texts)
            
        # Log the first example to help with debugging
        if isinstance(texts, list) and len(texts) > 0:
            logger.info(f"Sample text (first 100 chars): {texts[0][:100]}...")
        elif isinstance(texts, str):
            logger.info(f"Sample text (first 100 chars): {texts[:100]}...")
            
        # Tokenize with proper padding and truncation
        try:
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            logger.error(f"Text type: {type(texts)}")
            if isinstance(texts, list):
                logger.error(f"First text item type: {type(texts[0]) if texts else 'empty list'}")
            raise
        
        # For causal language modeling, we need input_ids and labels
        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        
        # For training, we also set labels equal to input_ids for causal LM
        if is_train:
            result["labels"] = tokenized["input_ids"].clone()
            
        return result
    
    # Apply the tokenization function to the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
    )
    
    logger.info(f"Preprocessed dataset has {len(tokenized_dataset)} examples")
    return tokenized_dataset

def setup_train_config(args):
    """
    Set up the training configuration based on the command-line arguments.
    """
    # Directly set the data path based on the phase
    if args.phase == 1:
        data_path = "phase1_data.txt"
    else:
        data_path = "phase2_data.txt"
    
    # Set up the training configuration
    train_config = TrainConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_path,
        dataset_path=data_path,
        phase=args.phase,
        output_dir=os.path.join(args.output_dir, f"phase{args.phase}"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size if args.batch_size > args.micro_batch_size else 1,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.1,  # As per OpenHathi parameters
        warmup_ratio=0.1,  # As per OpenHathi parameters
        lr_scheduler_type="cosine",
        context_length=args.cutoff_len,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        use_fp16=args.use_fp16,
        enable_fsdp=args.enable_fsdp,
        run_validation=True,
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        eval_steps=100,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    # Set up the FSDP configuration if enabled
    fsdp_config = FSDPConfig(
        pure_bf16=False,
        optimizer="adamw",
        fsdp_activation_checkpointing=True,
    )
    
    # Set up the quantization configuration
    quant_config = QuantizationConfig()
    
    # Set up the W&B configuration if enabled
    wandb_config = WandbConfig(
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"yanomami-phase{args.phase}",
        watch_model=False,
        save_model=False,
    ) if args.use_wandb else None
    
    return train_config, fsdp_config, quant_config, wandb_config

def main_cli():
    """
    Main function to set up and run the fine-tuning process.
    """
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set up the training configuration
    train_config, fsdp_config, quant_config, wandb_config = setup_train_config(args)
    
    # Log the configuration
    logger.info(f"Training configuration: {asdict(train_config)}")
    if args.enable_fsdp:
        logger.info(f"FSDP configuration: {asdict(fsdp_config)}")
    if args.use_wandb:
        logger.info(f"W&B configuration: {asdict(wandb_config)}")
    
    # Import necessary functions from the local llama_cookbook.py file
    try:
        logger.info("Importing required modules...")
        
        # Import from the local llama_cookbook.py file
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Current script directory: {current_dir}")
        
        # Check if llama_cookbook.py exists in the current directory
        cookbook_path = os.path.join(current_dir, 'llama_cookbook.py')
        if os.path.exists(cookbook_path):
            logger.info(f"Found local llama_cookbook.py at {cookbook_path}")
            
            # Import the main function directly from the local file
            import importlib.util
            spec = importlib.util.spec_from_file_location("llama_cookbook", cookbook_path)
            llama_cookbook_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llama_cookbook_module)
            
            # Get the main function from the local module
            llama_main = llama_cookbook_module.main
            logger.info("Successfully imported main function from local llama_cookbook.py")
        else:
            logger.error(f"Local llama_cookbook.py not found at {cookbook_path}")
            raise ImportError(f"Local llama_cookbook.py not found at {cookbook_path}")
        
        # Use datasets library directly instead of llama_cookbook.data
        from datasets import load_dataset, load_from_disk
        logger.info("Successfully imported datasets module.")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during module import: {str(e)}")
        sys.exit(1)
    
    # Check if the preprocessed dataset already exists
    preprocessed_path = os.path.join(os.path.dirname(train_config.dataset_path), "preprocessed")
    logger.info(f"Checking for preprocessed dataset at {preprocessed_path}")
    
    # Debug: Check if the data files exist
    logger.info(f"Checking if phase1 file exists: {os.path.exists(train_config.phase1_file)}")
    logger.info(f"Checking if phase2 file exists: {os.path.exists(train_config.phase2_file)}")
    
    # Debug: Print the current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    
    if os.path.exists(preprocessed_path):
        logger.info(f"Preprocessed dataset already exists at {preprocessed_path}. Skipping processing.")
        train_config.dataset_path = preprocessed_path
        logger.info(f"Updated dataset_path to {train_config.dataset_path}")
    else:
        # Load and preprocess the dataset
        logger.info(f"Loading dataset from {train_config.dataset_path}")
        
        try:
            # Directly load the dataset from the specified file
            logger.info(f"Attempting to load dataset for phase {train_config.phase}...")
            if train_config.phase == 1:
                logger.info(f"Loading phase 1 dataset from {train_config.phase1_file}")
                raw_dataset = load_dataset('text', data_files=train_config.phase1_file)
                logger.info(f"Successfully loaded phase 1 dataset: {raw_dataset}")
            else:
                logger.info(f"Loading phase 2 dataset from {train_config.phase2_file}")
                raw_dataset = load_dataset('text', data_files=train_config.phase2_file)
                logger.info(f"Successfully loaded phase 2 dataset: {raw_dataset}")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        
        # Load tokenizer
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {train_config.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.tokenizer_name or train_config.model_name,
            padding_side="right",
            use_fast=True,
        )
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Preprocess the dataset with proper padding and truncation
        preprocessed_dataset = preprocess_dataset(
            raw_dataset, 
            tokenizer, 
            max_length=train_config.context_length
        )
        
        # Save the preprocessed dataset to disk
        preprocessed_dataset.save_to_disk(preprocessed_path)
        
        # Update the dataset path to use the preprocessed dataset
        train_config.dataset_path = preprocessed_path
        logger.info(f"Saved preprocessed dataset to {preprocessed_path}")
        
        # Update the dataset path to point to the current directory
        original_dataset_path = train_config.dataset_path
        train_config.dataset_path = os.path.dirname(train_config.dataset_path)

        # Check if phase1_data.txt or phase2_data.txt exists in the current directory
        phase1_file = train_config.phase1_file
        phase2_file = train_config.phase2_file
        
        # Determine which phase file to use based on the specified phase
        phase_file = "phase1_data.txt" if train_config.phase == 1 else "phase2_data.txt"
        
        if not os.path.exists(phase_file):
            logger.error(f"Phase {train_config.phase} data file not found: {phase_file}")
            logger.error(f"Please make sure the file exists in {train_config.dataset_path}")
            sys.exit(1)
        
        # Convert the text file to a format that the llama_cookbook can use
        # We need to create a temporary JSON file in the expected format
        logger.info(f"Converting {phase_file} to the format expected by llama_cookbook...")
        
        # Create a directory structure that matches what the cookbook expects
        # First, create a formatted_data directory if it doesn't exist
        formatted_data_dir = os.path.join(os.path.dirname(phase_file), "formatted_data")
        os.makedirs(formatted_data_dir, exist_ok=True)
        
        # Create a JSON file with the expected format
        json_file = os.path.join(formatted_data_dir, "train.json")
        
        try:
            with open(phase_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Process the text file based on the phase
            if train_config.phase == 1:
                # Phase 1: Each entry should be a translated paragraph followed by English paragraph
                # Format: {"text": "<translated_text>\n\n<english_text>"}
                data = []
                logger.info("Processing Phase 1 data: translated paragraph followed by English paragraph")
                
                # Try to detect the format of the file
                # If each paragraph is on a single line and paragraphs are separated by blank lines
                if len([line for line in lines if line.strip()]) / 2 < len(lines) * 0.8:
                    # Format with blank lines between paragraphs
                    logger.info("Detected format: paragraphs separated by blank lines")
                    current_paragraph = []
                    for line in lines:
                        if line.strip():
                            current_paragraph.append(line.strip())
                        elif current_paragraph:  # Empty line and we have content
                            if len(current_paragraph) >= 2:
                                # Assuming first half is Yanomami, second half is English
                                mid = len(current_paragraph) // 2
                                yanomami_text = " ".join(current_paragraph[:mid])
                                english_text = " ".join(current_paragraph[mid:])
                                data.append({"text": f"{yanomami_text}\n\n{english_text}"})
                            current_paragraph = []
                    
                    # Don't forget the last paragraph if there's no final blank line
                    if current_paragraph and len(current_paragraph) >= 2:
                        mid = len(current_paragraph) // 2
                        yanomami_text = " ".join(current_paragraph[:mid])
                        english_text = " ".join(current_paragraph[mid:])
                        data.append({"text": f"{yanomami_text}\n\n{english_text}"})
                else:
                    # Assuming alternating lines: translated text followed by English text
                    logger.info("Detected format: alternating lines (Yanomami, English, Yanomami, ...)")
                    i = 0
                    while i < len(lines):
                        if i + 1 < len(lines):
                            yanomami_text = lines[i].strip()
                            english_text = lines[i+1].strip()
                            if yanomami_text and english_text:  # Skip empty lines
                                data.append({"text": f"{yanomami_text}\n\n{english_text}"})
                            i += 2
                        else:
                            i += 1
            else:
                # Phase 2: Alternating sentences in English and Yanomami
                # Format: {"text": "<sentence1_en> <sentence1_yanomami> <sentence2_en> ..."}
                logger.info("Processing Phase 2 data: alternating sentences in English and Yanomami")
                data = []
                
                # Try to detect if the file already has alternating sentences on each line
                # or if we need to process it differently
                for line in lines:
                    if line.strip():
                        # Just use each non-empty line as is
                        data.append({"text": line.strip()})
                
                logger.info(f"Processed {len(data)} lines for Phase 2 training")
            
            # Write the JSON file
            with open(json_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Created {json_file} with {len(data)} entries")
            
            # Update the dataset path to point to the formatted_data directory
            original_dataset_path = train_config.dataset_path
            train_config.dataset_path = formatted_data_dir
            
            # Update the output directory to be phase-specific
            original_output_dir = train_config.output_dir
            train_config.output_dir = os.path.join(original_output_dir, f"phase{train_config.phase}")
            os.makedirs(train_config.output_dir, exist_ok=True)
            
            logger.info(f"Updated dataset path from {original_dataset_path} to: {train_config.dataset_path}")
            logger.info(f"Updated output directory from {original_output_dir} to: {train_config.output_dir}")
            
        except Exception as e:
            logger.error(f"Error processing the data file: {str(e)}")
            sys.exit(1)
        
        # According to the cookbook, we need to use specific parameters for training
        # Based on the OpenHathi training parameters mentioned in the cookbook
        # and the parameters used in the llama-cookbook finetuning.py file
        
        # Start with the parameters specified in the cookbook
        kwargs = {
            # Model and tokenizer parameters
            "model_name": train_config.model_name,
            "tokenizer_name": train_config.tokenizer_name,
            "output_dir": train_config.output_dir,
            
            # Dataset parameters
            "dataset_path": train_config.dataset_path,  # Path to the formatted_data directory
            "dataset_format": "jsonl",  # Format used in the cookbook
            "dataset_text_field": "text",  # Field name in the JSON file
            "batching_strategy": "packing",  # From cookbook finetuning.py
            "context_length": 4096,  # From cookbook
            
            # Training parameters
            "lr": 2e-4,  # maximum learning rate from cookbook
            "weight_decay": 0.1,  # from cookbook
            "gamma": 0.85,  # LR decay factor
            "num_train_epochs": train_config.num_train_epochs,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "per_device_train_batch_size": train_config.per_device_train_batch_size,
            "per_device_eval_batch_size": train_config.per_device_eval_batch_size,
            "warmup_steps": 50,  # Number of warmup steps for learning rate scheduler
            "run_validation": True,  # Run validation during training
            "seed": train_config.seed,
            
            # PEFT (LoRA) parameters
            "use_peft": True,  # We need PEFT for LoRA
            "peft_method": "lora",  # Using LoRA as specified in cookbook
            "lora_r": 128,  # from cookbook
            "lora_alpha": 64,  # from cookbook
            "lora_dropout": 0.05,  # from cookbook
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],  # from cookbook
            
            # FSDP parameters (for distributed training)
            "enable_fsdp": False,  # Disable FSDP for single-GPU training
            "low_cpu_fsdp": False,
            "use_fast_kernels": True,  # Enable fast kernels for training
            "use_fp16": False,  # Use BF16 instead as recommended in cookbook
        }
        
        # Log the kwargs being passed to main
        logger.info(f"Passing the following parameters to llama_cookbook.finetuning.main: {kwargs.keys()}")
        
        # Remove parameters that might not be recognized
        # Only keep the necessary parameters for dataset processing
        for param in ["min_lr", "beta1", "beta2", "block_size", "dtype", "train"]:
            if param in kwargs:
                del kwargs[param]
        
        # Set environment variable for PyTorch memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        # Ensure PEFT is enabled to reduce memory usage
        kwargs['use_peft'] = True
        
        # Enable mixed precision for memory efficiency
        kwargs['mixed_precision'] = True
        
        # Run the fine-tuning process
        try:
            # First try with all parameters
            logger.info("Attempting to run with all parameters from the cookbook...")
            logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            result = llama_main(**kwargs)
        except (TypeError, RuntimeError) as e:
            logger.error(f"Error calling llama_main: {str(e)}")
            
            if isinstance(e, RuntimeError) and 'CUDA out of memory' in str(e):
                logger.error("CUDA out of memory error detected. Reducing context length and batch size...")
                
                # Reduce context length to save memory
                kwargs['context_length'] = 2048  # Reduce from 4096 to 2048
                logger.info(f"Reduced context_length to {kwargs['context_length']}")
                
                # Increase gradient accumulation to compensate for smaller batch size
                kwargs['gradient_accumulation_steps'] = 4
                logger.info(f"Increased gradient_accumulation_steps to {kwargs['gradient_accumulation_steps']}")
                
                # Clear cache again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache again")
                
                try:
                    logger.info(f"Retrying with reduced memory parameters: {kwargs.keys()}")
                    result = llama_main(**kwargs)
                except Exception as e2:
                    logger.error(f"Still encountering errors: {str(e2)}")
                    logger.error("Trying with minimal parameters...")
                    
                    # Try with minimal parameters focused on memory efficiency
                    minimal_kwargs = {
                        "model_name": train_config.model_name,
                        "tokenizer_name": train_config.tokenizer_name,
                        "output_dir": train_config.output_dir,
                        "dataset_path": train_config.dataset_path,
                        "use_peft": True,
                        "peft_method": "lora",
                        "lora_r": 64,  # Reduced from 128
                        "lora_alpha": 32,  # Reduced from 64
                        "context_length": 1024,  # Further reduced
                        "per_device_train_batch_size": 1,
                        "gradient_accumulation_steps": 8,
                        "mixed_precision": True,
                    }
                    logger.info(f"Trying with minimal parameters: {minimal_kwargs.keys()}")
                    result = llama_main(**minimal_kwargs)
            else:
                # Handle TypeError (parameter mismatch)
                logger.error("The function signature may have changed. Trying with a subset of parameters...")
                
                # Remove parameters that might not be recognized
                for param in ["min_lr", "beta1", "beta2", "block_size", "dtype"]:
                    if param in kwargs:
                        del kwargs[param]
                
                logger.info(f"Trying with reduced parameters: {kwargs.keys()}")
                result = llama_main(**kwargs)
        
        logger.info(f"Fine-tuning completed with result: {result}")
        
        return result

if __name__ == "__main__":
    try:
        main_cli()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
