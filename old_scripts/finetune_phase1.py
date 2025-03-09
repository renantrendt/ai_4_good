#!/usr/bin/env python3
"""
Script to fine-tune Llama on Phase 1 Yanomami language data.
Phase 1: Learn to translate paragraphs of text (use translated text as context and generate the original text).
"""

# Set tokenizer parallelism environment variable to avoid warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune_phase1.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer
from huggingface_hub import login

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama on Phase 1 Yanomami language data")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Meta-Llama-3.1-8B", 
        help="Base Llama model to fine-tune"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to use for training (for testing)"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="./extended_tokenizer",
        help="Path to the extended tokenizer"
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="./formatted_data/phase1_data.txt",
        help="Path to Phase 1 training data file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./llama-yanomami-phase1",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=4,
        help="Batch size per GPU for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4,
        help="Number of updates steps to accumulate before backward pass"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=128,
        help="LoRA attention dimension"
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
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--bf16", 
        action="store_true",
        help="Use bfloat16 precision if available"
    )
    parser.add_argument(
        "--use_4bit", 
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--deepspeed", 
        type=str, 
        default=None,
        help="Path to deepspeed config file"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        required=True,
        help="Hugging Face token for downloading models"
    )
    
    return parser.parse_args()

def load_phase1_dataset(train_file: str, max_samples: Optional[int] = None) -> Dataset:
    """Load and prepare Phase 1 dataset for training."""
    logger.info(f"Loading Phase 1 dataset from {train_file}")
    
    # Read the text file
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(lines)} lines from dataset file")
    except Exception as e:
        logger.error(f"Failed to read dataset file: {e}")
        raise
    
    # Create dataset entries
    dataset_entries = []
    for i, line in enumerate(lines):
        # Phase 1 format: "<yanomami text> = <english translation>"
        if "=" in line:
            try:
                yanomami_text, english_text = line.split("=", 1)
                # Store as a dictionary with a single text field
                dataset_entries.append({
                    "text": f"{yanomami_text.strip()}\n\n{english_text.strip()}"
                })
                if i < 3:  # Log a few examples
                    logger.debug(f"Sample {i}: {yanomami_text.strip()} = {english_text.strip()}")
            except Exception as e:
                logger.warning(f"Error processing line {i}: {e} - {line}")
    
    # Limit the number of samples if specified
    if max_samples is not None and max_samples > 0 and max_samples < len(dataset_entries):
        print(f"Limiting to {max_samples} samples for testing")
        dataset_entries = dataset_entries[:max_samples]
    
    print(f"Loaded {len(dataset_entries)} training examples")
    # Convert to Dataset from list of dictionaries
    return Dataset.from_list(dataset_entries)

def create_bnb_config(args):
    """Create BitsAndBytes configuration for quantization."""
    if args.use_4bit:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    return bnb_config

def create_peft_config(args):
    """Create PEFT configuration for LoRA."""
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "down_proj", "up_proj"
    ]
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return peft_config

def create_training_args(args):
    """Create training arguments for the Trainer."""
    logger.info("Creating training arguments with Llama cookbook recommended settings")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        save_strategy="epoch",
        logging_steps=1,  # Log more frequently for debugging
        report_to="tensorboard",
        # Let the data collator handle this - fix for CUDA indexing error
        remove_unused_columns=True,  
        push_to_hub=False,
        disable_tqdm=False,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        # Add warmup steps to avoid the 'learning rate scheduler' warning
        warmup_ratio=0.1,  # Following Llama cookbook recommendation
        # CUDA error fixes
        gradient_checkpointing=True,  # Save memory during training
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        # Set to empty since we're using our data collator to handle labels
        label_names=[],
        # Weight decay and optimizer settings to match DeepSpeed config
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,  # Changed to match DeepSpeed config
        # Add a comment to explain why we're using different beta values
        # Note: Using beta2=0.999 to match DeepSpeed config instead of 0.95 from Llama cookbook
    )
    
    logger.info(f"Training arguments created with batch size {args.per_device_train_batch_size} * {args.gradient_accumulation_steps} accumulation steps")
    return training_args

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Login to Hugging Face Hub
    if args.hf_token:
        login(token=args.hf_token)
    
    # Load the extended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the base model
    bnb_config = create_bnb_config(args)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=args.hf_token,
    )
    
    # Prepare model for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    peft_config = create_peft_config(args)
    model = get_peft_model(model, peft_config)
    
    # Enable input and output embedding training
    for param_name, param in model.named_parameters():
        if "embed_tokens" in param_name or "lm_head" in param_name:
            param.requires_grad = True
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load dataset
    train_dataset = load_phase1_dataset(args.train_file, args.max_samples)
    
    # Create training arguments
    training_args = create_training_args(args)
    
    # Create trainer
    # Following Llama cookbook for multilingual fine-tuning
    def tokenize_function(examples):
        logger.info(f"Tokenizing batch of {len(examples['text'])} examples")
        try:
            # Set return_tensors=None to avoid premature tensor conversion
            # that can cause dimension mismatches
            result = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=1024,  # Based on Llama cookbook recommended block size
                return_tensors=None,  # Let the data collator handle tensor conversion later
            )
            
            # Important: make sure input IDs and labels are properly set
            # This ensures the model gets correct labels during training
            result["labels"] = result["input_ids"].copy()
            
            logger.info(f"Successfully tokenized batch - input_ids count: {len(result['input_ids'])}")
            
            # Log sample token lengths to help with debugging
            token_lengths = [len(ids) for ids in result["input_ids"]]
            logger.info(f"Token length stats: min={min(token_lengths)}, max={max(token_lengths)}, avg={sum(token_lengths)/len(token_lengths):.1f}")
            
            # Check for any abnormally long or short sequences
            for i, length in enumerate(token_lengths):
                if length < 10 or length > 1000:
                    logger.warning(f"Example {i} has unusual token length: {length}")
                    
            return result
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            logger.error(f"Example text causing error: {examples['text'][0][:100]}...")
            raise
    
    # Apply the tokenization to the dataset
    logger.info("Starting dataset tokenization")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],  # Remove the original text column
        desc="Tokenizing dataset",
    )
    logger.info(f"Tokenization complete - Dataset size: {len(tokenized_dataset)}")
    
    # Log dataset structure to verify it's correctly formatted
    logger.info(f"Dataset structure: {tokenized_dataset.column_names}")
    logger.info(f"First example keys: {list(tokenized_dataset[0].keys())}")
    
    # Create a custom data collator for causal language modeling
    from transformers import DataCollatorForLanguageModeling
    logger.info("Creating DataCollatorForLanguageModeling")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    logger.info("Data collator created successfully")
    
    # Configure a standard Trainer instead of SFTTrainer
    # This avoids compatibility issues with SFTTrainer versions
    from transformers import Trainer
    logger.info("Initializing Trainer with model and dataset")
    
    # Add a custom compute_loss method to avoid CUDA indexing errors
    class LlamaTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            logger.debug(f"Computing loss with input keys: {inputs.keys()}")
            
            # Ensure we don't have any unexpected inputs that could cause CUDA errors
            if 'attention_mask' not in inputs:
                logger.warning("No attention_mask in inputs, creating a mask of ones")
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Log batch size information for debugging
            if num_items_in_batch is not None:
                logger.debug(f"Processing batch with {num_items_in_batch} items")
                
            # Standard transformer forward pass with labels
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels'],
                return_dict=True
            )
            
            # Return loss and outputs if needed
            return (outputs.loss, outputs) if return_outputs else outputs.loss
    
    try:
        trainer = LlamaTrainer(
            model=model,
            train_dataset=tokenized_dataset,
            args=training_args,
            data_collator=data_collator,
        )
        logger.info("Trainer initialized successfully with custom loss computation")
    except Exception as e:
        logger.error(f"Error initializing Trainer: {e}")
        raise
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(args.output_dir)
    
    # Create inference script
    create_inference_script(args.output_dir)
    
    print(f"Training complete. Model saved to {args.output_dir}")

def create_inference_script(model_dir):
    """Create a simple script for inference with the fine-tuned model."""
    script_path = os.path.join(model_dir, "inference.py")
    
    script_content = """#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load the model and tokenizer
model_path = "./"  # Current directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

def generate_text(input_text, max_length=100):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    # Phase 1 example: Yanomami to English
    yanomami_text = "hepisiprou"
    print(f"Input: {yanomami_text}")
    result = generate_text(yanomami_text)
    print(f"Output: {result}")
"""
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    print(f"Created inference script at {script_path}")

if __name__ == "__main__":
    main()