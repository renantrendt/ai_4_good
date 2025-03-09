#!/usr/bin/env python3
"""
Script to prepare Yanomami language data for tokenizer training.
This script samples documents from a dataset and saves them as text files.
"""

import os
import argparse
import random
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for tokenizer training")
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--lang", 
        type=str, 
        default="yan",
        help="Language code for the data (yan for Yanomami)"
    )
    parser.add_argument(
        "--docs_to_sample", 
        type=int, 
        default=10000,
        help="Number of documents to sample"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="./data",
        help="Directory to save the data"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default=None,
        help="HuggingFace dataset name (if using a public dataset)"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default=None,
        help="Path to local dataset files"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Load dataset
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name, split=args.split)
        print(f"Loaded {len(dataset)} examples from {args.dataset_name}")
    elif args.dataset_path:
        dataset = load_dataset("json", data_files=args.dataset_path, split=args.split)
        print(f"Loaded {len(dataset)} examples from {args.dataset_path}")
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")
    
    # Sample documents
    if len(dataset) > args.docs_to_sample:
        indices = random.sample(range(len(dataset)), args.docs_to_sample)
        dataset = dataset.select(indices)
    
    # Extract text from dataset and save to file
    output_file = os.path.join(args.save_path, f"{args.lang}.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc=f"Processing {args.lang} data"):
            # Extract text field - adjust this based on your dataset structure
            if "text" in example:
                text = example["text"]
            elif "content" in example:
                text = example["content"]
            elif "translation" in example and args.lang in example["translation"]:
                text = example["translation"][args.lang]
            else:
                # If you have a different structure, modify this part
                text = str(example)
            
            # Write to file
            f.write(text.strip() + "\n")
    
    print(f"Saved {len(dataset)} documents to {output_file}")

if __name__ == "__main__":
    main()