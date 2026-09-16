##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset preparation](#dataset-preparation)
3. [How to run](#how-to-run)
4. [Key hyper-parameters](#key-hyper-parameters)

# Location Matters: Sparse Latent-Variance Attacks for Face Privacy against Diffusion Personalization (SEAL)

SEAL protects a face image against Diffusion personalization by perturbing only ~5% of its pixels.

## Environment setup

```shell
cd SEAL
conda create -n seal python=3.9
conda activate seal
pip install -r requirements.txt
```

Pretrained Stable Diffusion checkpoints can be downloaded from the links below. Version 2.1 is the default; set `MODEL_PATH` in the scripts to a local snapshot or a HF repo id.
[Manojb/stable-diffusion-2-1-base](https://huggingface.co/Manojb/stable-diffusion-2-1-base)

The BiSeNet face-parsing checkpoint (from [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)) is already bundled at `model/79999_iter.pth`.

## Dataset preparation

We follow the Anti-DreamBooth protocol on CelebA-HQ and VGGFace2. The full splits are provided [here](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV).

## How to run

End-to-end demo (protect → DreamBooth → generate):

```shell
bash scripts/attack_with_seal.sh
```

Protection only:

```shell
python attacks/seal.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --instance_data_dir=data/CelebA-HQ/5/set_B \
  --output_dir=outputs/SEAL/CelebA-HQ/5/noise-ckpt
```

To train a DreamBooth model from your own data, whether clean or protected:

```shell
bash scripts/train_dreambooth_alone.sh
```

Inference: generates examples with multiple prompts

```shell
python infer.py --model_path <path to DREAMBOOTH model>/checkpoint-1000 --output_dir ./test-infer/
```

## Key hyper-parameters

All flags are documented inline in `scripts/attack_with_seal.sh`. 
Setting `--saliency_method=none` or `--lambda_cluster=0` disables the clustering penalty and reduces to a random sparse mask.

## Acknowledgement

The DreamBooth training and inference code is built on
[Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth) and
[diffusers](https://github.com/huggingface/diffusers); the face-parsing network comes from
[face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch).