# Yanomami Language Extension for Llama 3.1

## Project Overview

**You can test the model at: https://yanomami.bernardoserrano.com**

This project aims to extend the Meta Llama 3.1 model to support the Yanomami language, an indigenous language spoken by approximately 35,000 people in the Amazon rainforest regions of Brazil and Venezuela. By fine-tuning Llama 3.1 on Yanomami language data, we create a multilingual model capable of understanding and generating text in both Yanomami and English.

1. We trained a model Llama 3.1 8B INT8 using 8xA100 GPUs on Lambdalabs
2. We created a chat interface for the model using assistant-ui https://github.com/renantrendt/yanomami-chat
3. We are hosting the model on Lambdalabs for inference
4. We are adding Qdrant to the model for knowledge retrieval
5. We plan to create an app that runs offline because on the forest there is no internet connection


The project follows the Meta Llama cookbook approach for extending language models to new languages, implementing a two-phase training process:

1. **Phase 1**: Learning to translate paragraphs (translated text as context, generate original text)
2. **Phase 2**: Bilingual next token prediction (alternating sentences in both languages)

### Phase 1: Translation Learning

**Objective**: Teach the model to understand the relationship between the new language (Yanomami) and English.

**Data Format**: `{"text": "<translated_text>\n\n<english_text>"}`
- The model is given translated text as context and learns to generate the original English text.
- Example: Yanomami text followed by two newlines, then the corresponding English text.

**Learning Focus**: 
- Basic vocabulary and grammar of the new language
- Mapping between concepts in both languages
- Understanding the structure of the new language

**File Used**: `formatted_data/phase1_data.txt`

### Phase 2: Bilingual Next Token Prediction

**Objective**: Improve the model's ability to seamlessly switch between languages and generate coherent text in both.

**Data Format**: `{"text": "<sentence1_en> <sentence1_yanomami> <sentence2_en> ..."}`
- Alternating sentences in both languages
- The model learns to predict the next token regardless of which language it's in

**Learning Focus**:
- Code-switching (moving between languages)
- Maintaining context across language boundaries
- Generating coherent text in both languages

**File Used**: `formatted_data/phase2_data.txt`

This two-phase approach is designed to gradually build the model's capabilities in the new language, first establishing basic understanding and translation abilities, then developing more sophisticated bilingual capabilities.

## Installation Instructions

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (training was performed on 8xA100 GPUs)
- Hugging Face account with access to Llama 3.1 models

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/llama-yanomami-extension.git
   cd llama-yanomami-extension
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   sudo apt-get update
   sudo apt-get install -y build-essential python3-dev
   ```

3. Set up environment variables:
   ```bash
   export HF_DATASETS_TRUST_REMOTE_CODE=True
   ```

4. Move necessary files to the correct locations:
   ```bash
   # If using llama_cookbook
   mv extend_language/samsum_dataset.py ~/.local/lib/python3.10/site-packages/llama_cookbook/datasets/
   ```

## Usage Guide

### Data Preparation

1. Prepare your dataset:
   ```bash
   python prepare_data.py --dataset_path ./dataset/validation.jsonl --save_path ./data
   ```
   
2. (Optional) Set up SamSum dataset for testing and benchmarking:
   ```bash
   # Clone the SamSum dataset repository
   git clone https://huggingface.co/datasets/Samsung/samsum
   
   # Set environment variable for remote code execution
   export HF_DATASETS_TRUST_REMOTE_CODE=True
   
   # Process the SamSum dataset
   python extend_language/samsum/samsum.py
   
   # Run the SamSum dataset preparation script
   python /home/ubuntu/.local/lib/python3.10/site-packages/llama_cookbook/datasets/samsum_dataset.py
   ```

3. Train a tokenizer for Yanomami:
   ```bash
   python train_tokenizer.py --input_file ./data/yan.txt --save_path ./yanomami_tokenizer --vocab_size 8000
   ```

4. Extend the base Llama tokenizer with Yanomami tokens:
   ```bash
   python extend_tokenizer_v2.py \ 
       --base_model_name meta-llama/Meta-Llama-3.1-8B \ 
       --new_tokenizer_path ./yanomami_tokenizer \ 
       --extended_tokenizer_save_path ./extended_tokenizer \ 
       --hf_token YOUR_HF_TOKEN
   ```

5. Fix the tokenizer (to address 'OrderedVocab contains holes' warning):
   ```bash
   python fix_tokenizer.py --tokenizer_path ./extended_tokenizer --output_path ./fixed_tokenizer
   ```

6. Prepare training data for both phases:
   ```bash
   # Phase 1: Translation format
   python prepare_training_data.py --input_file ./dataset/train.jsonl --output_dir ./formatted_data --phase 1
   
   # Phase 2: Bilingual next token prediction format
   python prepare_training_data.py --input_file ./dataset/train.jsonl --output_dir ./formatted_data --phase 2
   ```

### Training

#### Phase 1: Translation Learning

1. Quick test with a small sample:
   ```bash
   python finetune_phase1.py --max_samples 20 \ 
       --model_name meta-llama/Meta-Llama-3.1-8B \ 
       --tokenizer_path ./fixed_tokenizer \ 
       --train_file ./formatted_data/phase1_data.txt \ 
       --output_dir ./llama-yanomami-phase1 \ 
       --bf16 \ 
       --use_4bit \ 
       --num_train_epochs 3 \ 
       --per_device_train_batch_size 4 \ 
       --gradient_accumulation_steps 4 \ 
       --learning_rate 2e-4 \ 
       --lora_r 128 \ 
       --lora_alpha 64 \ 
       --deepspeed ds_config_zero2.json \ 
       --hf_token YOUR_HF_TOKEN
   ```

2. Full training with all samples:
   ```bash
   python finetune_phase1.py \ 
       --model_name meta-llama/Meta-Llama-3.1-8B \ 
       --tokenizer_path ./fixed_tokenizer \ 
       --train_file ./formatted_data/phase1_data.txt \ 
       --output_dir ./llama-yanomami-phase1 \ 
       --bf16 \ 
       --use_4bit \ 
       --num_train_epochs 3 \ 
       --per_device_train_batch_size 4 \ 
       --gradient_accumulation_steps 4 \ 
       --learning_rate 2e-4 \ 
       --max_seq_length 4096 \ 
       --lora_r 128 \ 
       --lora_alpha 64 \ 
       --deepspeed ds_config_zero2.json \ 
       --hf_token YOUR_HF_TOKEN
   ```

#### Phase 2: Bilingual Next Token Prediction

```bash
python finetune_phase2.py \ 
    --model_name ./llama-yanomami-phase1 \ 
    --tokenizer_path ./extended_tokenizer \ 
    --train_file ./formatted_data/phase2_data.txt \ 
    --output_dir ./llama-yanomami-phase2 \ 
    --bf16 \ 
    --use_4bit \ 
    --num_train_epochs 3 \ 
    --per_device_train_batch_size 4 \ 
    --gradient_accumulation_steps 4 \ 
    --learning_rate 2e-4 \ 
    --max_seq_length 4096 \ 
    --lora_r 128 \ 
    --lora_alpha 64 \ 
    --deepspeed ds_config_zero2.json \ 
    --hf_token YOUR_HF_TOKEN
```

### Using the Unified Script

Alternatively, you can use our unified script that follows the Meta Llama cookbook approach:

```bash
python yanomami_finetune.py --phase=1 --use_peft
```

## Model Implementation Details

This project implements the Meta Llama cookbook approach for extending language models to new languages. Key components include:

### Data Format

- **Phase 1**: `{"text": "<translated_text>\n\n<english_text>"}`
- **Phase 2**: `{"text": "<sentence1_en> <sentence1_yanomami> <sentence2_en> ..."}`

### Training Parameters

Following the cookbook recommendations:

- Learning rate: 2e-4
- LoRA rank: 128
- LoRA alpha: 64
- LoRA target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj
- Context length: 4096
- Training hardware: 8xA100 GPUs
- Mixed precision: BF16
- Quantization: 4-bit (QLoRA)
- DeepSpeed Zero-2 optimization

### Memory Optimization

The training process uses several memory optimization techniques:

- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- 4-bit quantization (QLoRA)
- DeepSpeed Zero-2 for distributed training
- Gradient accumulation
- Mixed precision training

## Ethical Considerations and Limitations

### Ethical Considerations

1. **Cultural Preservation**: This project contributes to the digital preservation of the Yanomami language, supporting linguistic diversity and cultural heritage.

2. **Informed Consent**: Ensure that any Yanomami language data used has been collected with proper informed consent from native speakers and communities.

3. **Representation**: The model should accurately represent Yanomami language and culture without perpetuating stereotypes or misrepresentations.

4. **Access**: Consider how to make the resulting model accessible to Yanomami communities who could benefit from it.

### Limitations

1. **Data Scarcity**: Limited availability of high-quality Yanomami language data may affect model performance.

2. **Cultural Nuance**: The model may not capture all cultural nuances and contextual meanings specific to Yanomami culture.

3. **Dialect Variation**: The Yanomami language has several dialects, and the model may not represent all of them equally.

4. **Technical Requirements**: The computational resources required for inference may limit accessibility in remote areas where many Yanomami communities are located.

5. **Evaluation Challenges**: Limited availability of native Yanomami speakers for model evaluation may affect quality assessment.

## Acknowledgments

This project follows the approach outlined in the [Meta Llama Cookbook](https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/multilingual/README.md) for extending language models to new languages.

## License

This project is licensed under [LICENSE TYPE] - see the LICENSE file for details.
