#!/usr/bin/env python3
"""
Script to train a SentencePiece tokenizer for the Yanomami language.
This script trains a tokenizer using the prepared text data.
"""

import os
import argparse
from pathlib import Path
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast

def parse_args():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer for Yanomami")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Path to the input text file"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=8000,
        help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--character_coverage", 
        type=float, 
        default=1.0,
        help="Character coverage for the tokenizer"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="bpe",
        choices=["bpe", "unigram", "char", "word"],
        help="SentencePiece model type"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="./yanomami_tokenizer",
        help="Directory to save the tokenizer"
    )
    return parser.parse_args()

def train_tokenizer(args):
    """
    Train a SentencePiece tokenizer using the provided arguments.
    """
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Set the model prefix
    model_prefix = os.path.join(args.save_path, "tokenizer")
    
    # Train the SentencePiece model
    spm.SentencePieceTrainer.train(
        input=args.input_file,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        # Additional parameters
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        normalization_rule_name="nmt_nfkc_cf"
    )
    
    print(f"SentencePiece model trained and saved to {model_prefix}.model and {model_prefix}.vocab")
    
    # Load the trained model
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(f"{model_prefix}.model")
    
    # Convert to Hugging Face format
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{model_prefix}.model",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    
    # Save the tokenizer in Hugging Face format
    tokenizer.save_pretrained(args.save_path)
    print(f"Tokenizer saved in Hugging Face format to {args.save_path}")
    
    return tokenizer

def test_tokenizer(tokenizer, input_file):
    """
    Test the trained tokenizer on a sample from the input file.
    """
    # Read a sample from the input file
    with open(input_file, "r", encoding="NFC") as f:
        lines = f.readlines()
        sample_text = lines[0].strip() if lines else "No sample text available"
    
    # Tokenize the sample text
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)
    
    print("\nTokenizer test:")
    print(f"Sample text: {sample_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")
    
    # Print vocabulary statistics
    vocab_size = tokenizer.vocab_size
    print(f"\nVocabulary size: {vocab_size}")
    
    # Print some example tokens from the vocabulary
    print("\nSample vocabulary items:")
    for i in range(min(10, vocab_size)):
        print(f"ID {i}: {tokenizer.convert_ids_to_tokens(i)}")

def main():
    args = parse_args()
    
    print(f"Training tokenizer with vocabulary size {args.vocab_size}...")
    tokenizer = train_tokenizer(args)
    
    print("\nTesting tokenizer...")
    test_tokenizer(tokenizer, args.input_file)
    
    print("\nTokenizer training complete!")

if __name__ == "__main__":
    main()