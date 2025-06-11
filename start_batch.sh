#!/bin/bash
cd "/root/Grad"
source venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/root/Grad:$PYTHONPATH"

# 如果有HF token，设置环境变量
if [ -n "" ]; then
    export HUGGINGFACE_HUB_TOKEN=""
fi

# 运行命令
exec python3 knowledge_coupling_batch_processor.py --model_path 'meta-llama/Llama-2-7b-hf' --batch_size 2000 --sub_batch_size 10 --checkpoint_dir '/root/Grad/checkpoints' --output_dir '/root/Grad/results/full_hotpotqa_analysis' --layer_start 28 --layer_end 31 --no_resume 2>&1 | tee -a "/root/Grad/logs/batch_processing.log"
