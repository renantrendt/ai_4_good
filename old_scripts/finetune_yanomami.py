#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Llama models on Yanomami language data.
This script uses the standalone llama_cookbook.py implementation.

Usage:
    python finetune_yanomami.py --phase=1 --model_name="meta-llama/Meta-Llama-3.1-8B" --tokenizer_path="./fixed_tokenizer"
"""

import argparse
import os
import logging
from dataclasses import asdict

import torch
from transformers import set_seed

from llama_cookbook import (
    TrainConfig, 
    FSDPConfig, 
    QuantizationConfig, 
    WandbConfig,
    main
)

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

def setup_train_config(args):
    """
    Set up the training configuration based on the command-line arguments.
    """
    # Determine data path based on phase
    data_path = os.path.join(args.data_path, f"phase{args.phase}")
    
    # Set up the training configuration
    train_config = TrainConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_path,
        dataset_path=data_path,
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
    
    # Run the fine-tuning process
    result = main(
        train_config=train_config,
        fsdp_config=fsdp_config if args.enable_fsdp else None,
        quantization_config=quant_config,
        wandb_config=wandb_config if args.use_wandb else None,
    )
    
    logger.info(f"Fine-tuning completed with result: {result}")
    
    return result

if __name__ == "__main__":
    main_cli()
