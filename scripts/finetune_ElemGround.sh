#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed --master_port 14218 --num_nodes=1 --num_gpus=8 llava_uhd/train/ui_llava/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/jingran_su/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298 \
    --version llama_3 \
    --data_path /data0/jingran/workspace/UI_training_data/Ours-Pretrain/mixed_AG_point_tag.json \
    --image_folder /data0/jingran/workspace/UI_training_data/Ours-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-llama3-8b_pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-llama3-8b_mixed_AG_tag \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    # --add_point_token True # Add <point> </point> to special tokens
