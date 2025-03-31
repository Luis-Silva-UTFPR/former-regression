#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set environment variables if necessary
export PYTHONPATH=$(pwd)

# Run the finetuning script with the required arguments
python finetuning.py \
    --dataset_path "10_10_unique_mun.parquet" \
    --finetune_path "../checkpoints/finetune" \
    --pretrain_path "../checkpoints/pretrain" \
    --num_workers 16 \
    --max_length 95 \
    --patch_size 10 \
    --num_classes 1 \
    --hidden_size 256 \
    --attn_heads 8 \
    --learning_rate 1e-5 \
    --weight_decay 1e-3 \
    --epochs 800 \
    --batch_size 256 \
    --dropout 0.1
    

# python finetuning.py \
#     --dataset_path "10_10.parquet" \
#     --finetune_path "../checkpoints/finetune" \
#     --num_workers 16 \
#     --max_length 95 \
#     --patch_size 10 \
#     --num_classes 1 \
#     --hidden_size 256 \
#     --attn_heads 8 \
#     --learning_rate 1e-5 \
#     --weight_decay 1e-4 \
#     --epochs 400 \
#     --batch_size 256 \
#     --dropout 0.1
    
