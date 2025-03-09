#!/usr/bin/env python3
"""
Script to extend the Llama tokenizer with Yanomami tokens.
This script merges the Yanomami tokenizer with the Llama tokenizer,
ensuring Yanomami-specific tokens are prioritized.
"""

import os
import sys
import argparse
import sentencepiece as spm
import json
from transformers import AutoTokenizer, PreTrainedTokenizerFast

def parse_args():
    parser = argparse.ArgumentParser(description="Extend Llama tokenizer with Yanomami tokens")
    parser.add_argument(
        "--base_model_name", 
        type=str, 
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Base Llama model name or path"
    )
    parser.add_argument(
        "--new_tokenizer_path", 
        type=str, 
        required=True,
        help="Path to the new Yanomami tokenizer directory"
    )
    parser.add_argument(
        "--extended_tokenizer_save_path", 
        type=str, 
        default="./extended_tokenizer",
        help="Directory to save the extended tokenizer"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None,
        help="Hugging Face token for downloading models"
    )
    parser.add_argument(
        "--priority_factor", 
        type=float, 
        default=1.5,
        help="Factor to increase priority of Yanomami tokens (higher means more priority)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def get_yanomami_samples():
    """Return a list of Yanomami text samples for testing"""
    return [
        "hepisiprou warorai hokiã itahi",
        "Yanomami thëpë urihipë ha, kama yama ki kupru tëhë, ɨhɨ tëhë yamakɨ xaari thaɨ",
        "Hwei thë pata thëpë ã haɨ tëhë, kami yamakɨ urihipë ha, yamakɨ xaari kuo",
        "Yanomami thëpë urihipë ha, kama yama ki kupru tëhë, ɨhɨ tëhë yamakɨ xaari thaɨ"
    ]

def modify_tokenizer_json(tokenizer_path, yan_vocab, priority_factor=1.5):
    """Modify the tokenizer.json file to prioritize Yanomami tokens"""
    tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        print(f"Warning: tokenizer.json not found at {tokenizer_json_path}")
        return False
    
    try:
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Check if the tokenizer has the expected structure
        if 'model' not in tokenizer_data or 'vocab' not in tokenizer_data['model']:
            print("Warning: tokenizer.json doesn't have the expected structure")
            return False
        
        # Increase scores for Yanomami tokens to prioritize them
        modified = False
        for token, _ in yan_vocab.items():
            if token in tokenizer_data['model']['vocab']:
                # Increase the score (lower is better in BPE)
                current_score = tokenizer_data['model']['vocab'][token]
                # For BPE tokenizers, lower scores mean higher priority
                new_score = int(current_score / priority_factor)
                tokenizer_data['model']['vocab'][token] = new_score
                modified = True
        
        if modified:
            # Save the modified tokenizer
            with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            print(f"Modified tokenizer.json to prioritize Yanomami tokens")
            return True
        else:
            print("No Yanomami tokens found in the tokenizer vocabulary")
            return False
    except Exception as e:
        print(f"Error modifying tokenizer.json: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    try:
        print("Starting tokenizer extension process...")
        args = parse_args()
        
        # Create save directory if it doesn't exist
        os.makedirs(args.extended_tokenizer_save_path, exist_ok=True)
        print(f"Created directory: {args.extended_tokenizer_save_path}")
        
        # Load the base Llama tokenizer using AutoTokenizer
        print(f"Loading base tokenizer from {args.base_model_name}...")
        try:
            # Fix the token handling issue
            if args.hf_token:
                base_tokenizer = AutoTokenizer.from_pretrained(
                    args.base_model_name,
                    use_fast=True,
                    token=args.hf_token
                )
            else:
                base_tokenizer = AutoTokenizer.from_pretrained(
                    args.base_model_name,
                    use_fast=True
                )
            print("Base tokenizer loaded successfully.")
            print(f"Base tokenizer type: {type(base_tokenizer).__name__}")
        except Exception as e:
            print(f"Error loading base tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Load the Yanomami tokenizer
        print(f"Loading Yanomami tokenizer from {args.new_tokenizer_path}...")
        try:
            # First check if we have the tokenizer.json file (HF format)
            tokenizer_json_path = os.path.join(args.new_tokenizer_path, "tokenizer.json")
            if os.path.exists(tokenizer_json_path):
                yan_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.new_tokenizer_path)
                print("Yanomami tokenizer loaded from HF format successfully.")
                print(f"Yanomami tokenizer type: {type(yan_tokenizer).__name__}")
                
                # Get the Yanomami vocabulary
                yan_vocab = yan_tokenizer.get_vocab()
            else:
                # Fall back to loading from SentencePiece model
                yan_model_path = os.path.join(args.new_tokenizer_path, "tokenizer.model")
                if not os.path.exists(yan_model_path):
                    print(f"Error: Yanomami tokenizer model not found at {yan_model_path}")
                    return
                
                yan_sp = spm.SentencePieceProcessor()
                yan_sp.load(yan_model_path)
                print("Yanomami tokenizer loaded from SentencePiece model successfully.")
                
                # Extract vocabulary from SentencePiece model
                yan_vocab = {}
                for i in range(yan_sp.get_piece_size()):
                    piece = yan_sp.id_to_piece(i)
                    yan_vocab[piece] = i
        except Exception as e:
            print(f"Error loading Yanomami tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Get the base vocabulary
        base_vocab = base_tokenizer.get_vocab()
        base_vocab_size = len(base_vocab)
        print(f"Base vocabulary size: {base_vocab_size}")
        print(f"Yanomami vocabulary size: {len(yan_vocab)}")
        
        # Find new tokens to add (tokens in Yanomami vocab but not in base vocab)
        new_tokens = []
        for token in yan_vocab:
            if token not in base_vocab:
                new_tokens.append(token)
        
        print(f"Number of new tokens to add: {len(new_tokens)}")
        
        # Add new tokens to the base tokenizer
        if new_tokens:
            num_added = base_tokenizer.add_tokens(new_tokens)
            print(f"Added {num_added} new tokens to the base tokenizer")
        else:
            print("Warning: No new tokens were added. This suggests the tokenizer extension may not be effective.")
        
        print(f"Extended vocabulary size: {len(base_tokenizer)}")
        
        # Save the extended tokenizer
        base_tokenizer.save_pretrained(args.extended_tokenizer_save_path)
        print(f"Extended tokenizer saved to {args.extended_tokenizer_save_path}")
        
        # Modify the tokenizer.json to prioritize Yanomami tokens
        if modify_tokenizer_json(args.extended_tokenizer_save_path, yan_vocab, args.priority_factor):
            print("Successfully modified tokenizer to prioritize Yanomami tokens")
        
        # Test the extended tokenizer
        try:
            print("\nTesting the extended tokenizer...")
            # Use AutoTokenizer instead of LlamaTokenizer
            test_tokenizer = AutoTokenizer.from_pretrained(args.extended_tokenizer_save_path)
            print(f"Test tokenizer type: {type(test_tokenizer).__name__}")
            
            # Get Yanomami samples for testing
            samples = get_yanomami_samples()
            
            # Compare tokenization for each sample
            for i, sample_text in enumerate(samples):
                print(f"\nSample {i+1}: {sample_text}")
                
                # Compare tokenization
                base_tokens = base_tokenizer.tokenize(sample_text)
                extended_tokens = test_tokenizer.tokenize(sample_text)
                
                print(f"Base tokenizer ({len(base_tokens)} tokens): {base_tokens}")
                print(f"Extended tokenizer ({len(extended_tokens)} tokens): {extended_tokens}")
                
                # Calculate token reduction percentage
                if len(base_tokens) > 0 and len(base_tokens) != len(extended_tokens):
                    reduction = (len(base_tokens) - len(extended_tokens)) / len(base_tokens) * 100
                    if reduction > 0:
                        print(f"Token reduction: {reduction:.2f}% (more efficient)")
                    else:
                        print(f"Token increase: {-reduction:.2f}% (more detailed)")
            
            # Verify that the original tokens are preserved
            print("\nVerifying original tokens are preserved...")
            for i in range(min(10, base_vocab_size)):
                base_token = base_tokenizer.convert_ids_to_tokens(i)
                extended_token = test_tokenizer.convert_ids_to_tokens(i)
                if base_token != extended_token:
                    print(f"Token mismatch at index {i}: {base_token} != {extended_token}")
                    break
            else:
                print("Sample of original tokens are preserved in the extended tokenizer")
                
        except Exception as e:
            print(f"Error testing extended tokenizer: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print("Script execution completed.")