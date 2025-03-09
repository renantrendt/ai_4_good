mv /home/ubuntu/extend_language/samsum_dataset.py /home/ubuntu/.local/lib/python3.10/site-packages/llama_cookbook/datasets/

Install instance: scp -r /Users/renanserrano/CascadeProjects/Yanomami/finetune-nllb/ ssh ubuntu@129.146.102.17:~/

pip instal -r requirements.txt
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

```
python3 prepare_data.py --dataset_path ./dataset/validation.jsonl --save_path ./data
````
```
python train_tokenizer.py --input_file ./data/yan.txt --save_path ./yanomami_tokenizer --vocab_size 8000
```

```
python extend_tokenizer_v2.py --base_model_name meta-llama/Meta-Llama-3.1-8B --new_tokenizer_path ./yanomami_tokenizer --extended_tokenizer_save_path ./extended_tokenizer --hf_token YOUR_HF_TOKEN
```

# Fix the tokenizer (to address 'OrderedVocab contains holes' warning)
```
python fix_tokenizer.py --tokenizer_path ./extended_tokenizer --output_path ./fixed_tokenizer
```


```
python prepare_training_data.py --input_file ./dataset/train.jsonl --output_dir ./formatted_data --phase 1

```
python prepare_training_data.py --input_file ./dataset/train.jsonl --output_dir ./formatted_data --phase 2
```
```
git clone https://huggingface.co/datasets/Samsung/samsum

export HF_DATASETS_TRUST_REMOTE_CODE=True

python extend_language/samsum/samsum.py

python /home/ubuntu/.local/lib/python3.10/site-packages/llama_cookbook/datasets/samsum_dataset.py
````


````
# Quick test with a small sample
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

```
# Full training with all samples
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

````

````
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


1. We have the @Llama-Guard-3-8B-int8 model downloaded in the repository.
2. use the tool to analyze those 4 pages: https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning & https://github.com/huggingface/huggingface-llama-recipes/blob/main/fine_tune/peft_finetuning.py & https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/multilingual/README.md & https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/finetuning.py
3. The objective is to extend Llama to a new language called Yanomami. 
4. We will build a training script, you need to follow the insctructions from the links that I sent to you, mainly from the multilingual.
5. We will use 8xA100 GPUS