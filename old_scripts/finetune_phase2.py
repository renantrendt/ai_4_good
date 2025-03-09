#!/usr/bin/env python3
"""
Script to fine-tune Llama on Phase 2 Yanomami language data.
Phase 2: Bilingual next token prediction (train on text where the language changes after every sentence).
"""

import os
import torch
import argparse
from pathlib import Path
from typing import Dict, Optional

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer
from huggingface_hub import login

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama on Phase 2 Yanomami language data")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Path to the Phase 1 fine-tuned model"
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
        default="./formatted_data/phase2_data.txt",
        help="Path to Phase 2 training data file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./llama-yanomami-phase2",
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
        "--max_seq_length", 
        type=int, 
        default=4096,
        help="Maximum sequence length for training"
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

def load_phase2_dataset(train_file: str) -> Dataset:
    """Load and prepare Phase 2 dataset for training."""
    print(f"Loading Phase 2 dataset from {train_file}")
    
    # Read the text file
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into paragraphs (separated by double newlines)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    # Create dataset entries
    dataset_entries = []
    for paragraph in paragraphs:
        dataset_entries.append({"text": paragraph})
    
    print(f"Loaded {len(dataset_entries)} training examples")
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
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        save_strategy="epoch",
        logging_steps=10,
        report_to="tensorboard",
        remove_unused_columns=False,
        push_to_hub=False,
        disable_tqdm=False,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
    )
    
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
    
    # Load the Phase 1 fine-tuned model
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
    train_dataset = load_phase2_dataset(args.train_file)
    
    # Create training arguments
    training_args = create_training_args(args)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    
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
    # Phase 2 example: Bilingual text
    english_text = "to become thin at a point."
    print(f"Input: {english_text}")
    result = generate_text(english_text)
    print(f"Output: {result}")
    
    # Try with Yanomami text
    yanomami_text = "hepisiprou."
    print(f"\\nInput: {yanomami_text}")
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