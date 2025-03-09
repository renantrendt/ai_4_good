#!/usr/bin/env python3
"""
Script to prepare training data for the two-phase approach to fine-tune Llama for Yanomami.
Phase 1: Create data where Yanomami text is followed by its English translation.
Phase 2: Create bilingual data where the language alternates after each sentence.
"""

import os
import json
import argparse
import random
from tqdm import tqdm
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training data for the two-phase approach")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Path to the input JSONL file (train.jsonl or validation.jsonl)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./training_data",
        help="Directory to save the formatted training data"
    )
    parser.add_argument(
        "--phase", 
        type=int, 
        choices=[1, 2],
        required=True,
        help="Phase 1 or Phase 2 data preparation"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def extract_yanomami_english_pairs(jsonl_file):
    """
    Extract Yanomami-English word/phrase pairs from the dictionary JSONL file.
    """
    pairs = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            messages = data.get('messages', [])
            
            if len(messages) < 2:
                continue
                
            user_msg = messages[0].get('content', '')
            assistant_msg = messages[1].get('content', '')
            
            # Extract the Yanomami word
            word_match = re.search(r'<WORD>(.*?)</WORD>', assistant_msg)
            if not word_match:
                continue
            
            yanomami_word = word_match.group(1)
            
            # Extract the definition (English translation)
            definition_match = re.search(r'<DEFINITION>(.*?)</DEFINITION>', assistant_msg)
            if not definition_match:
                continue
                
            english_definition = definition_match.group(1)
            
            # Extract examples if available
            examples = []
            example_pattern = re.compile(r'<EXAMPLE_YANOMAMI>(.*?)</EXAMPLE_YANOMAMI>.*?<EXAMPLE_TRANSLATION>(.*?)</EXAMPLE_TRANSLATION>', re.DOTALL)
            for yan_ex, eng_ex in example_pattern.findall(assistant_msg):
                if yan_ex.strip() and eng_ex.strip():
                    examples.append((yan_ex.strip(), eng_ex.strip()))
            
            # Add the word-definition pair
            pairs.append({
                'yanomami': yanomami_word,
                'english': english_definition,
                'examples': examples
            })
    
    return pairs

def create_phase1_data(pairs, output_file):
    """
    Create Phase 1 data: Yanomami text followed by its English translation.
    Format: "<yanomami text> = <english translation>"
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in tqdm(pairs, desc="Creating Phase 1 data"):
            # Write word-definition pair
            f.write(f"{pair['yanomami']} = {pair['english']}\n")
            
            # Write examples
            for yan_ex, eng_ex in pair['examples']:
                f.write(f"{yan_ex} = {eng_ex}\n")
            
            f.write("\n")  # Add a blank line between entries

def create_phase2_data(pairs, output_file):
    """
    Create Phase 2 data: Bilingual text where sentences alternate between Yanomami and English.
    This format helps the model learn to switch between languages seamlessly.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in tqdm(pairs, desc="Creating Phase 2 data"):
            # Process examples for alternating sentences
            for yan_ex, eng_ex in pair['examples']:
                # Skip if either example is empty
                if not yan_ex.strip() or not eng_ex.strip():
                    continue
                    
                # Split into sentences (simple split by punctuation)
                yan_sentences = [s.strip() for s in re.split(r'[.!?]', yan_ex) if s.strip()]
                eng_sentences = [s.strip() for s in re.split(r'[.!?]', eng_ex) if s.strip()]
                
                # Ensure we have enough sentences to alternate
                if not yan_sentences or not eng_sentences:
                    continue
                
                # Create alternating text
                alternating_sentences = []
                
                # Use the minimum length to ensure we have pairs
                min_length = min(len(yan_sentences), len(eng_sentences))
                
                # Create alternating sentences
                for i in range(min_length):
                    # Add English sentence first, then Yanomami
                    alternating_sentences.append(f"{eng_sentences[i]}.")
                    alternating_sentences.append(f"{yan_sentences[i]}.")
                
                # Join sentences with spaces
                alternating_text = " ".join(alternating_sentences)
                
                # Write the alternating text
                f.write(f"{alternating_text}\n\n")
            
            # Also create alternating text from the word-definition pair
            if pair['yanomami'] and pair['english']:
                f.write(f"{pair['english']}. {pair['yanomami']}.\n\n")

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract Yanomami-English pairs from the JSONL file
    print(f"Extracting Yanomami-English pairs from {args.input_file}...")
    pairs = extract_yanomami_english_pairs(args.input_file)
    print(f"Extracted {len(pairs)} Yanomami-English pairs")
    
    # Determine output file based on phase
    phase_name = f"phase{args.phase}"
    output_file = os.path.join(args.output_dir, f"{phase_name}_data.txt")
    
    # Create the formatted data based on the specified phase
    if args.phase == 1:
        create_phase1_data(pairs, output_file)
    else:  # Phase 2
        create_phase2_data(pairs, output_file)
    
    print(f"Created {phase_name} data and saved to {output_file}")

if __name__ == "__main__":
    main()