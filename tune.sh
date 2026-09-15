#!/bin/bash
#SBATCH --job-name=LLMs
#SBATCH --partition=priogp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log

source ~/research/sentiment_analysis/venv/bin/activate

cd ~/research/sentiment_analysis

python lib/finetune_model.py \
    --model_path /nlsasfs/home/aidrive/dassuv/models/Llama-3.1-8B-Instruct \
    --data_csv /nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/files_from_rajvee/LID_with_consolidated.csv \
    --output_dir /nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/lid_checkpoints/ \
    --batch_size 8 