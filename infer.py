import argparse
import os

import torch
from diffusers import StableDiffusionPipeline
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from PIL import Image
import numpy as np
from einops import rearrange

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="The output directory where predictions are saved",
)
parser.add_argument(
    "--v",
    type=str,
    default="sks",
    help="The output directory where predictions are saved",
)
parser.add_argument(
    "--num_images_per_prompt",
    type=int,
    default=32,
    help="Total images to generate per prompt.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Mini-batch size per pipeline call (to bound VRAM).",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="fp16",
    choices=["fp16", "fp32"],
    help="Inference dtype.",
)

args = parser.parse_args()


def _run_infer(args):
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # define prompts
    prompts = [
        f"a photo of {args.v} person",
        f"a dslr portrait of {args.v} person",
        f"a photo of {args.v} person looking at the mirror",
        f"a photo of {args.v} person in front of eiffel tower",
        f"a close-up photo of {args.v} person, high details",
        f"a photo of {args.v} person stand in front of the table holding a knife",
    ]


    # create & load model
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        local_files_only=True,
    ).to("cuda")
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    total = args.num_images_per_prompt
    bs = max(1, min(args.batch_size, total))

    for prompt in prompts:
        print(">>>>>>", prompt)
        norm_prompt = prompt.lower().replace(",", "").replace(" ", "_")
        out_path = f"{args.output_dir}/{norm_prompt}"
        os.makedirs(out_path, exist_ok=True)
        all_samples = list()
        generated = 0
        idx = 0
        while generated < total:
            cur = min(bs, total - generated)
            images = pipe(
                [prompt] * cur,
                num_inference_steps=100,
                guidance_scale=7.5,
            ).images
            for image in images:
                image.save(f"{out_path}/{idx}.png")
                arr = np.array(image, dtype=np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                all_samples.append(torch.from_numpy(arr))
                idx += 1
            generated += cur
            torch.cuda.empty_cache()
        grid = torch.stack(all_samples, 0)
        grid = make_grid(grid, nrow=4)
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8))
        img.save(f"{args.output_dir}/{prompt}.png")
        torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    _run_infer(args)
