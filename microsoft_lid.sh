#!/bin/bash
#SBATCH --job-name=microsoft_lid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log

source ~/research/sentiment_analysis/venv/bin/activate

cd ~/research/sentiment_analysis

python3 lib/microsoft_lid.py 