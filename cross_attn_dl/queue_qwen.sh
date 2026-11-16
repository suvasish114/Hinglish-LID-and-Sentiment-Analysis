#!/bin/bash
# Wait until both GPUs have enough free memory for Qwen2.5-7B (bf16 weights sharded 0:7GiB,1:20GiB + fp32 copies of
# the last 2 blocks + optimiser), then run the whole pipeline for it. Launch detached:
#   setsid nohup bash queue_qwen.sh > logs/queue_qwen.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
NEED0=${NEED0:-10000}; NEED1=${NEED1:-22000}      # MiB free required on GPU 0 / GPU 1
while true; do
  read -r f0 f1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')
  if [ "$f0" -ge "$NEED0" ] && [ "$f1" -ge "$NEED1" ]; then
    echo "$(date) free=${f0}/${f1} MiB -> launching Qwen pipeline"
    break
  fi
  echo "$(date) free=${f0}/${f1} MiB, need ${NEED0}/${NEED1}, waiting"
  sleep 120
done
EXTRA='--dtype bfloat16 --device_map auto --max_memory 0:7GiB,1:20GiB --gradient_checkpointing --eval_batch_size 32' \
  bash run.sh Qwen/Qwen2.5-7B-Instruct > logs/pipeline_Qwen2.5-7B-Instruct.log 2>&1
echo "$(date) Qwen pipeline exited with $?"
