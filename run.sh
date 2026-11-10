#!/bin/bash
#SBATCH --job-name=cs613
#SBATCH --partition=priogp
#SBATCH --nodelist=scn83-10g
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
# proxy for Internet access
export http_proxy="http://proxy-10g.10g.siddhi.param:9090 "
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export ftp_proxy= "http://proxy-10g.10g.siddhi.param:9090"
export http_proxy="http://proxy-10g.10g.siddhi.param:9090"
export https_proxy="$http_proxy"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
cd $SLURM_SUBMIT_DIR
source ~/research/sentiment_analysis/venv/bin/activate
cd ~/research/sentiment_analysis
DATA_DIR="/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/dataset/SentiMix"
python3 routing_cross_attention.py Qwen/Qwen2.5-7B-Instruct \
    --model-path "/nlsasfs/home/aidrive/dassuv/models/Qwen2.5-7B-Instruct" \
    --train-csv "$DATA_DIR/train.csv" \
    --dev-csv "$DATA_DIR/dev.csv" \
    --test-csv "$DATA_DIR/test.csv" \
    --device cuda