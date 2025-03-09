#!/usr/bin/env python3
"""
Script to fix the extended tokenizer by ensuring continuous token IDs without holes.
This addresses the 'OrderedVocab contains holes' warning.
"""

import os
import json
import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Fix extended tokenizer to remove holes in vocabulary")
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="./extended_tokenizer",
        help="Path to the extended tokenizer directory"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./fixed_tokenizer",
        help="Path to save the fixed tokenizer"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def fix_tokenizer_json(tokenizer_path, output_path, verbose=False):
    """Fix the tokenizer.json file to ensure continuous token IDs without holes"""
    tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        print(f"Error: tokenizer.json not found at {tokenizer_json_path}")
        return False
    
    try:
        # Load the tokenizer.json file
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Check if the tokenizer has the expected structure
        if 'model' not in tokenizer_data or 'vocab' not in tokenizer_data['model']:
            print("Error: tokenizer.json doesn't have the expected structure")
            return False
        
        # Get the vocabulary and sort by token ID
        vocab = tokenizer_data['model']['vocab']
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        
        # Create a new vocabulary with continuous IDs
        new_vocab = {}
        for i, (token, _) in enumerate(sorted_vocab):
            new_vocab[token] = i
        
        # Replace the old vocabulary with the new one
        tokenizer_data['model']['vocab'] = new_vocab
        
        # Update the merges if present (for BPE tokenizers)
        if 'merges' in tokenizer_data['model']:
            # Sort merges by their implicit rank
            merges = tokenizer_data['model']['merges']
            # We don't need to modify merges as they don't contain IDs
        
        # Save the modified tokenizer
        os.makedirs(output_path, exist_ok=True)
        output_json_path = os.path.join(output_path, "tokenizer.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        # Copy other tokenizer files
        for filename in os.listdir(tokenizer_path):
            if filename != "tokenizer.json":
                src_path = os.path.join(tokenizer_path, filename)
                dst_path = os.path.join(output_path, filename)
                if os.path.isfile(src_path):
                    with open(src_path, 'rb') as src_file:
                        with open(dst_path, 'wb') as dst_file:
                            dst_file.write(src_file.read())
        
        print(f"Fixed tokenizer saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error fixing tokenizer.json: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    try:
        print("Starting tokenizer fixing process...")
        args = parse_args()
        
        # Fix the tokenizer
        if fix_tokenizer_json(args.tokenizer_path, args.output_path, args.verbose):
            print("Successfully fixed tokenizer vocabulary to ensure continuous IDs")
        else:
            print("Failed to fix tokenizer")
            return
        
        # Test the fixed tokenizer
        try:
            print("\nTesting the fixed tokenizer...")
            original_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            fixed_tokenizer = AutoTokenizer.from_pretrained(args.output_path)
            
            # Compare vocabulary sizes
            orig_vocab = original_tokenizer.get_vocab()
            fixed_vocab = fixed_tokenizer.get_vocab()
            print(f"Original vocabulary size: {len(orig_vocab)}")
            print(f"Fixed vocabulary size: {len(fixed_vocab)}")
            
            # Test on a sample text
            sample_text = "Yanomami thëpë urihipë ha, kama yama ki kupru tëhë, ɨhɨ tëhë yamakɨ xaari thaɨ"
            
            # Compare tokenization
            orig_tokens = original_tokenizer.tokenize(sample_text)
            fixed_tokens = fixed_tokenizer.tokenize(sample_text)
            
            print(f"\nSample text: {sample_text}")
            print(f"Original tokenizer ({len(orig_tokens)} tokens): {orig_tokens}")
            print(f"Fixed tokenizer ({len(fixed_tokens)} tokens): {fixed_tokens}")
            
            # Check for holes in the fixed tokenizer
            max_id = max(fixed_vocab.values())
            ids_set = set(fixed_vocab.values())
            if len(ids_set) != max_id + 1:
                missing_ids = [i for i in range(max_id + 1) if i not in ids_set]
                print(f"Warning: Fixed tokenizer still has holes at indices: {missing_ids}")
            else:
                print("Fixed tokenizer has continuous IDs without holes")
            
        except Exception as e:
            print(f"Error testing fixed tokenizer: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print("Script execution completed.")
