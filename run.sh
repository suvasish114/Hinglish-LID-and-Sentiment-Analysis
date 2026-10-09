#!/bin/bash
#SBATCH --job-name=mist1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log

source ~/research/sentiment_analysis/venv/bin/activate

cd ~/research/sentiment_analysis

# python3 main.py --model_id "Qwen/Qwen2.5-7B-Instruct" \
# python3 main.py --model_id "CohereLabs/aya-expanse-8b" \
# python3 main.py --model_id "meta-llama/Llama-3.1-8B-Instruct" \
python3 main.py --model_id "mistralai/Mistral-7B-Instruct-v0.3" \
    --output_path "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/results" \
    --shot_value 1 \
    --test_data  "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/comi_lingua_dataset/test.csv" \
    --max_seq_length 256 \
    --temperature 0.0 \
    --zero_shot_prompt_path "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/prompts/zero_shot_lid_prompt.txt" \
    --one_shot_prompt_path "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/prompts/one_shot_lid_prompt.txt" \
    --index_from 1 \
    --index_to 1000
