#!/usr/bin/env bash
### Train DreamBooth on an arbitrary image folder (clean or protected),
### then run inference. Useful for the clean baseline: point INSTANCE_DIR
### at data/CelebA-HQ/5/set_B instead of the SEAL output.

set -e

export CUDA_VISIBLE_DEVICES=0

### Third-party mirror of stabilityai/stable-diffusion-2-1-base, used because
### the official repo was unreachable at the time these experiments were run.
export MODEL_PATH="Manojb/stable-diffusion-2-1-base"
export EXPERIMENT_NAME="clean/CelebA-HQ/5"
export CLASS_DIR="data/class-person"
export INSTANCE_DIR="data/CelebA-HQ/5/set_B"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/$EXPERIMENT_NAME"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompts="a photo of sks person;a dslr portrait of sks person;a photo of sks person looking at the mirror;a photo of sks person in front of eiffel tower;a close-up photo of sks person, high details;a photo of sks person stand in front of the table holding a knife" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8 \
  --seed=42

python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000 \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer
