#!/usr/bin/env python3
"""
Script to fine-tune Llama-3.1-8B-Instruct on Yanomami language data using 8xA100 GPUs.
Optimized for distributed training on high-performance hardware.
"""

import os
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
from huggingface_hub import login

# Add system message to each example if not present
def preprocess_dataset(dataset):
    """Add system message to each example if not present."""
    processed_dataset = []
    system_message = {"role": "system", "content": "You are a helpful assistant specialized in Yanomami language translation and dictionary lookup."}
    
    for example in dataset:
        messages = example["messages"]
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, system_message)
        processed_dataset.append({"messages": messages})
    
    return Dataset.from_list(processed_dataset)

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.1-8B-Instruct on Yanomami language data")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
        help="Llama 3.1 model to fine-tune"
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="training_data/llama3/train.jsonl",
        help="Path to training data file"
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default="training_data/llama3/validation.jsonl",
        help="Path to validation data file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./llama3-yanomami-finetuned",
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
        default=8,
        help="Batch size per GPU for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=8,
        help="Batch size per GPU for evaluation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1,
        help="Number of updates steps to accumulate before backward pass"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=2048,
        help="Maximum sequence length for training"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
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
        "--fp16", 
        action="store_true",
        help="Use float16 precision"
    )
    parser.add_argument(
        "--use_4bit", 
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use_8bit", 
        action="store_true",
        help="Use 8-bit quantization"
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
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None,
        help="Hugging Face token for downloading models"
    )
    
    return parser.parse_args()

def load_datasets(train_file: str, validation_file: str) -> Dict[str, Dataset]:
    """Load and prepare datasets for training and validation."""
    print(f"Loading datasets from {train_file} and {validation_file}")
    
    # Load datasets
    datasets = {}
    if os.path.exists(train_file):
        datasets["train"] = load_dataset("json", data_files=train_file, split="train")
        print(f"Loaded {len(datasets['train'])} training examples")
    
    if os.path.exists(validation_file):
        datasets["validation"] = load_dataset("json", data_files=validation_file, split="train")
        print(f"Loaded {len(datasets['validation'])} validation examples")
    
    return datasets

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
        print("Using 4-bit quantization with BitsAndBytes")
    elif args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization with BitsAndBytes")
    else:
        bnb_config = None
        print("Using full precision (no quantization)")
    
    return bnb_config

def create_peft_config(args):
    """Create PEFT configuration for LoRA."""
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(f"Using LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    return peft_config

def create_training_args(args):
    """Create training arguments for the Trainer."""
    # Determine precision
    bf16_ready = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
        bf16=args.bf16 and bf16_ready,
        fp16=args.fp16 and not (args.bf16 and bf16_ready),
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        optim="adamw_torch",
    )
    
    print(f"Using {training_args.device} for training")
    if training_args.bf16:
        print("Using bfloat16 precision")
    elif training_args.fp16:
        print("Using float16 precision")
    else:
        print("Using float32 precision")
    
    return training_args

def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Login to Hugging Face Hub if token is provided
    if args.hf_token:
        login(token=args.hf_token)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    datasets = load_datasets(args.train_file, args.validation_file)
    
    # Preprocess datasets to add system message
    if "train" in datasets:
        datasets["train"] = preprocess_dataset(datasets["train"])
        print(f"Added system message to {len(datasets['train'])} training examples")
    
    if "validation" in datasets:
        datasets["validation"] = preprocess_dataset(datasets["validation"])
        print(f"Added system message to {len(datasets['validation'])} validation examples")
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Create BitsAndBytes config for quantization
    bnb_config = create_bnb_config(args)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        use_fast=True,
    )
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        use_cache=False,  # Disable KV cache for training
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 for faster training
        low_cpu_mem_usage=True,
    )
    
    # Prepare model for k-bit training if using quantization
    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Create PEFT config for LoRA
    peft_config = create_peft_config(args)
    
    # Create training arguments
    training_args = create_training_args(args)
    
    # Create SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("validation"),
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        dataset_text_field="messages",
        packing=False,  # Don't pack sequences for chat format
    )
    
    # Train model
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Create inference script
    create_inference_script(args.output_dir)
    
    print("Training complete!")

def create_inference_script(model_dir):
    """Create a simple script for inference with the fine-tuned model."""
    script_path = os.path.join(model_dir, "translate.py")
    script_content = '''
#!/usr/bin/env python3

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Translate text using fine-tuned Llama 3.1 model")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--model_dir", type=str, default="./", help="Directory containing the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum output length")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    config = PeftConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, args.model_dir)
    
    # Prepare input
    query = f"<QUERY>What does <WORD>{args.text}</WORD> mean in Yanomami?</QUERY>"
    messages = [{"role": "user", "content": query}]
    
    # Format input for the model
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate response
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_length,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    
    # Decode and print response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Input: {args.text}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    print(f"Created inference script at {script_path}")

if __name__ == "__main__":
    main()