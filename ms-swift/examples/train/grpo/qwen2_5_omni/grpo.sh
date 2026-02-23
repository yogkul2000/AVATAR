export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DECORD_EOF_RETRY_MAX=20480
export NNODES=1
export NODE_RANK=0
export MASTER_PORT=29501
export NPROC_PER_NODE=4
export MAX_PIXELS=401408
export ENABLE_AUDIO_OUTPUT=0 
export USE_AUDIO_IN_VIDEO=1 
export VIDEO_MAX_PIXELS=50176 
export FPS_MAX_FRAMES=16 

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Omni-7B \
    --reward_funcs custom_accuracy_reward custom_format_reward \
    --reward_weights 1 0.5 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "<path_to_stage_1_2_3_data>" \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200000000 \
    --save_steps 250 \
    --save_total_limit 2000000 \
    --logging_steps 1 \
    --logging_dir avatar/logs \
    --max_length 16384 \
    --output_dir avatar_grpo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 6 \
    --num_generations 8 \
    --steps_per_generation 8 \
    --top_p 1.0 \
    --top_k 50 \
    --beta 0.1 \
    --temperature 1. \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --attn_impl flash_attn \
    --tas_enable \
    --tas_off_policy_batches 4 \
    --tas_on_policy_batches 4 \
    --tas_off_policy_alpha 1.0 \
    --tas_lambda 0.3 \
    --tas_replay_buffer_size 10000 \
    --tas_replay_min_size 10 \
    --tas_replay_easy_ratio 0.25 \
    --tas_replay_medium_ratio 0.35 \
    --tas_replay_hard_ratio 0.40 \
    --tas_vcrs_window 20 \
    --tas_hint_key hint \
    --tas_hint_zero_patience 3 \
    --tas_replay_strategy priority
