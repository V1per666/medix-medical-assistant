#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct 

echo "Running DAPO training with model path: $MODEL_PATH"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=../data/medix-rl-data/data@train \
    data.val_files=../data/medix-rl-data/data@test \
    data.max_prompt_length=4352 \
    data.max_response_length=4096 \
    data.format_prompt=./examples/format_prompt/medical_format.jinja \
    data.answer_key=solution \
    data.image_key=image \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/medical.py:compute_score \
    worker.rollout.max_num_batched_tokens=8448 \
    data.rollout_batch_size=16 \
    data.mini_rollout_batch_size=16 \
    worker.actor.global_batch_size=16 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    trainer.experiment_name=medix-r1_2b_dapo \
    trainer.total_epochs=2 \
    trainer.val_generations_to_log=15 \
    trainer.val_before_train=False \
    worker.rollout.n=3 \
    trainer.save_checkpoint_path=./checkpoints/medix-r1_2b_dapo \
    trainer.n_gpus_per_node=2
