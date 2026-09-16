
### Hyper-parameters below are the defaults used in all reported experiments.

set -e

export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="Manojb/stable-diffusion-2-1-base"

export EXPERIMENT_NAME=5
export CLEAN_TRAIN_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_A"
export CLEAN_ADV_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_B"
export OUTPUT_DIR="outputs/SEAL/CelebA-HQ/$EXPERIMENT_NAME"
export CLASS_DIR="data/class-person"

mkdir -p $OUTPUT_DIR
cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean
cp -r $CLEAN_ADV_DIR   $OUTPUT_DIR/image_before_adding_noise

export NOISE_DIR="$OUTPUT_DIR/noise-ckpt"
mkdir -p $NOISE_DIR

# ------------------------- 1. Protect -------------------------
#   --attack_type   latent-distribution objective (laplace_loss = variance collapse)
#   --eps           L_inf budget in /255
#   --alpha         PGD step as a fraction of eps
#   --beta          mask-logit step as a fraction of sqrt(H*W)
#   --sparsity_ratio  L0 budget: fraction of pixels allowed to change
#   --patience      re-init delta/mask after this many steps with a frozen support
#
# Sensitivity map (drives the clustering penalty only; the support is still
# chosen by the sPGD top-k mask):
#   --saliency_method      bisenet | face | spectral | none
#   --bisenet_checkpoint   required when saliency_method=bisenet
#
# Local-clustering penalty:
#   --cluster_kernel_size        K of the K x K mean density kernel
#   --cluster_density_threshold  penalize where local_density * sensitivity
#                                exceeds this. An isolated pixel gives
#                                density = 1/K^2 = 0.020 (K=7), far below 0.15.
#   --lambda_cluster             direction-blending ratio between the unit VaC
#                                gradient and the unit penalty gradient on
#                                mask_logits (0 = pure VaC, 1.0 = equal weight).
#                                NOT an absolute loss weight: both gradients are
#                                unit-normalized first to cancel the ~1e9 scale
#                                gap between the VaC loss and the penalty.
python attacks/seal.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --instance_data_dir=$CLEAN_ADV_DIR \
  --output_dir=$NOISE_DIR \
  --resolution=512 \
  --center_crop \
  --max_train_steps=300 \
  --eps=12.75 \
  --alpha=0.25 \
  --beta=0.25 \
  --sparsity_ratio=0.05 \
  --patience=3 \
  --attack_type=laplace_loss \
  --lambda_constraint=0.1 \
  --saliency_method=bisenet \
  --bisenet_checkpoint="model/79999_iter.pth" \
  --cluster_kernel_size=7 \
  --cluster_density_threshold=0.15 \
  --lambda_cluster=0.5 \
  --seed=0

# ------------------------- 2. Train DreamBooth on the protected images ----------
export INSTANCE_DIR="$NOISE_DIR"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/SEAL/CelebA-HQ/$EXPERIMENT_NAME"

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

# ------------------------- 3. Inference ---------------------------------------
python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000 \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer
