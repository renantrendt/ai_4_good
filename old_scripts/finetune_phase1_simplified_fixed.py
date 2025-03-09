#!/usr/bin/env python3
"""
Simplified script to fine-tune Llama on Phase 1 Yanomami language data.
Phase 1: Learn to translate paragraphs of text (use translated text as context and generate the original text).
Following the Llama cookbook approach for multilingual fine-tuning.
"""

# Set tokenizer parallelism environment variable to avoid warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import argparse
import torch
from pathlib import Path
from typing import Optional

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

# Import necessary libraries
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    PreTrainedTokenizerFast
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
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
        default="./extended_tokenizer_sentencepiece",
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
        "--max_seq_length", 
        type=int, 
        default=1024,
        help="Maximum sequence length for training"
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
        logger.info(f"Limiting to {max_samples} samples for testing")
        dataset_entries = dataset_entries[:max_samples]
    
    # Create a Dataset object
    dataset = Dataset.from_list(dataset_entries)
    logger.info(f"Loaded {len(dataset)} training examples")
    
    return dataset

def create_bnb_config(use_4bit: bool):
    """Create BitsAndBytes configuration for quantization."""
    if use_4bit:
        logger.info("Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None
    
    return bnb_config

def create_peft_config(lora_r: int, lora_alpha: int, lora_dropout: float):
    """Create PEFT configuration for LoRA."""
    # Target modules based on Llama cookbook recommendations
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "down_proj", "up_proj"
    ]
    
    logger.info(f"Creating LoRA config with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return peft_config

def main():
    # Parse command-line arguments
    args = parse_args()
    set_seed(args.seed)
    
    # Login to Hugging Face Hub
    logger.info(f"Logging in to Hugging Face Hub with token")
    login(token=args.hf_token)
    
    # Load the extended tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    
    # Check if tokenizer is loaded correctly
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(f"Failed to load tokenizer from {args.tokenizer_path}")
    
    # Make sure we have a proper padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.info("Setting pad_token to a default value")
            tokenizer.pad_token = "</s>"  # Common EOS token for many models
    
    # Load the base model with quantization if specified
    logger.info(f"Loading base model {args.model_name}")
    bnb_config = create_bnb_config(args.use_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Print model parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
    
    # Apply LoRA
    logger.info("Applying LoRA to the model")
    peft_config = create_peft_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, peft_config)
    
    # Load the Phase 1 dataset
    train_dataset = load_phase1_dataset(args.train_file, args.max_samples)
    
    # Tokenize the dataset
    logger.info("Tokenizing the dataset")
    def tokenize_function(examples):
        # Tokenize with padding and truncation
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors=None,  # Return Python lists, not tensors
        )
        # Set labels equal to input_ids for causal language modeling
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")
    
    # Create training arguments
    logger.info("Creating training arguments")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # Following Llama cookbook recommendations
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # Training settings
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16,
        # DeepSpeed integration
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        # Memory optimization
        gradient_checkpointing=True,
        # Avoid multiprocessing issues
        dataloader_num_workers=0,
    )
    
    # Create data collator for language modeling
    logger.info("Creating data collator for language modeling")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
    )
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # Create inference script
    create_inference_script(args.output_dir)
    
    logger.info("Training complete!")

def create_inference_script(model_dir: str):
    """Create a simple script for inference with the fine-tuned model."""
    script_path = Path(model_dir) / "inference.py"
    logger.info(f"Creating inference script at {script_path}")
    
    script_content = '''
#!/usr/bin/env python3
"""Inference script for the fine-tuned Llama model on Yanomami language."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the model and tokenizer
def load_model(model_path):
    # Load the PEFT configuration
    peft_config = PeftConfig.from_pretrained(model_path)
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load the fine-tuned model
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load the tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    
    return model, tokenizer

# Generate text
def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Main function
def main():
    model_path = "./"  # Path to the model directory
    model, tokenizer = load_model(model_path)
    
    # Example usage
    prompt = "Enter your Yanomami text here"  # Replace with actual text
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content.strip())
    
    # Make the script executable
    script_path.chmod(script_path.stat().st_mode | 0o111)

if __name__ == "__main__":
    main()
