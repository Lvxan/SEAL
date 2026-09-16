"""
Usage:

    python attacks/seal.py \
      --pretrained_model_name_or_path=$MODEL_PATH \
      --instance_data_dir=data/CelebA-HQ/5/set_B \
      --output_dir=outputs/SEAL/CelebA-HQ/5/noise-ckpt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import AutoencoderKL

from saliency_module import (
    compute_bisenet_maps,
    get_saliency_map,
    load_bisenet,
)


# --------------------------------------------------------------------------- #
# Sparse masking
# --------------------------------------------------------------------------- #
class SparseMasking(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = mask.shape
        mask_flat = mask.detach().clone().view(b, -1)
        _, idx = torch.sort(mask_flat, dim=1, descending=True)
        mask_proj = torch.zeros_like(mask_flat).scatter_(1, idx[:, :k], 1).view(b, c, h, w)
        ctx.save_for_backward(x, mask)
        return x * mask_proj, mask_proj

    @staticmethod
    def backward(ctx, grad_output, _grad_mask_proj):
        x, mask = ctx.saved_tensors
        grad_mask = grad_output * x
        if mask.shape[1] == 1 and grad_mask.shape[1] != 1:
            grad_mask = grad_mask.mean(dim=1, keepdim=True)
        return grad_output * mask, grad_mask, None


def project_mask_topk(mask, k):
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    b, c, h, w = mask.shape
    flat = mask.view(b, -1)
    _, idx = torch.sort(flat, dim=1, descending=True)
    out = torch.zeros_like(flat).scatter_(1, idx[:, :k], 1).view(b, c, h, w)
    return out


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def laplace_loss(poison_var, mu=0.0, b=1.0):
    return torch.sum(torch.abs(poison_var - mu) / b)


def lagrange_entropy_loss(poison_var, epsilon=1e-8, lambda_constraint=0.1):
    entropy = -torch.sum(poison_var * torch.log(poison_var + epsilon))
    constraint = lambda_constraint * (torch.sum(poison_var) - 1) ** 2
    return entropy + constraint


def compute_loss(attack_type, clean_latent, poison_latent, lambda_constraint=0.1):
    if attack_type == "var":
        return F.mse_loss(clean_latent.std, poison_latent.std, reduction="mean")
    if attack_type == "mean":
        return F.mse_loss(clean_latent.mean, poison_latent.mean, reduction="mean")
    if attack_type == "KL":
        sigma_2, mu_2 = poison_latent.std, poison_latent.mean
        sigma_1, mu_1 = clean_latent.std, clean_latent.mean
        kl = (
            torch.log(sigma_2 / sigma_1)
            - 0.5
            + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2 ** 2)
        )
        return kl.flatten().mean()
    if attack_type == "latent_vector":
        return F.mse_loss(clean_latent.sample(), poison_latent.sample(), reduction="mean")
    if attack_type == "add":
        return F.mse_loss(clean_latent.std, poison_latent.std, reduction="mean") + F.mse_loss(
            clean_latent.mean, poison_latent.mean, reduction="mean"
        )
    if attack_type == "add-log":
        return F.mse_loss(
            clean_latent.var.log(), poison_latent.var.log(), reduction="mean"
        ) + F.mse_loss(clean_latent.mean, poison_latent.mean, reduction="mean")
    if attack_type == "laplace_loss":
        return laplace_loss(poison_latent.var)
    if attack_type == "lagrange_entropy_loss":
        return lagrange_entropy_loss(poison_latent.var, lambda_constraint=lambda_constraint)
    raise ValueError(f"unknown attack_type: {attack_type}")


# --------------------------------------------------------------------------- #
# Local dispersion penalty.
# --------------------------------------------------------------------------- #
def compute_cluster_penalty(
    mask_proj_diff: torch.Tensor,
    sensitivity_map: torch.Tensor,
    kernel: torch.Tensor,
    density_threshold: float,
) -> torch.Tensor:
    """
    mask_proj_diff:   (1, 1, H, W) binary-in-value, differentiable-in-grad.
    sensitivity_map:  (1, 1, H, W) in [0, 1], high = perceptually salient.
    kernel:           (1, 1, K, K) averaging kernel pre-built on the right device.
    Returns a scalar penalty (>= 0).
    """
    K = kernel.shape[-1]
    local_density = F.conv2d(mask_proj_diff, kernel, padding=K // 2)
    cluster_in_salient = local_density * sensitivity_map
    return F.relu(cluster_in_salient - density_threshold).mean()


# --------------------------------------------------------------------------- #
# Dataset.
# --------------------------------------------------------------------------- #
class InstanceImageDataset(Dataset):
    def __init__(self, instance_data_root, size=512, center_crop=False):
        self.size = size
        self.instance_images_path = sorted(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        img = Image.open(self.instance_images_path[index])
        img = exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return {"index": index, "pixel_values": self.image_transforms(img)}


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="SEAL: sparse perceptually-aware latent unlearnable examples."
    )
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--instance_data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--center_crop", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    # loss
    p.add_argument(
        "--attack_type",
        default="laplace_loss",
        choices=[
            "var", "mean", "KL", "latent_vector", "add", "add-log",
            "laplace_loss", "lagrange_entropy_loss",
        ],
    )
    p.add_argument("--lambda_constraint", type=float, default=0.1)

    # Sparse / PGD
    p.add_argument("--max_train_steps", type=int, default=300)
    p.add_argument("--eps", type=float, default=12.75)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--sparsity_ratio", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=3)

    # Sensitivity map (drives the local-clustering penalty only)
    p.add_argument("--saliency_method", type=str, default="bisenet",
                   choices=["face", "spectral", "bisenet", "none"])
    p.add_argument("--bisenet_checkpoint", type=str, default="model/79999_iter.pth")

    # Local clustering penalty (this file's contribution)
    p.add_argument("--cluster_kernel_size", type=int, default=7,
                   help="K of the K x K mean kernel for local density.")
    p.add_argument("--cluster_density_threshold", type=float, default=0.15,
                   help="Only penalize windows whose density * sensitivity exceeds this.")
    p.add_argument("--lambda_cluster", type=float, default=0.5,
                   help="Weight of the clustering penalty in total loss "
                        "(total = loss - lambda_cluster * penalty).")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Per-image attack.
# --------------------------------------------------------------------------- #
def attack_one_image(
    vae, x, args, k, device, weight_dtype, generator,
    sensitivity_map_img, cluster_kernel, image_idx=0,
):
    """
    x:                    (1, C, H, W) in [0, 1]
    sensitivity_map_img:  (1, 1, H, W) in [0, 1], or None to disable penalty.
    cluster_kernel:       (1, 1, K, K) averaging kernel on `device`.
    """
    x = x.to(device=device, dtype=weight_dtype)
    _, c, h, w = x.shape
    eps = args.eps / 255.0
    alpha = eps * args.alpha
    mask_step = float(np.sqrt(h * w)) * args.beta

    use_cluster = sensitivity_map_img is not None and args.lambda_cluster > 0
    if use_cluster:
        sens = sensitivity_map_img.to(device=device, dtype=weight_dtype)
    else:
        sens = None

    # ---- Clean latent reference ----
    with torch.no_grad():
        clean_latent = vae.encode(2 * x - 1).latent_dist
        clean_mean = clean_latent.mean.detach()
        clean_std = clean_latent.std.detach()
        clean_var = clean_latent.var.detach()

    class _CleanLatent:
        pass

    clean_ref = _CleanLatent()
    clean_ref.mean = clean_mean
    clean_ref.std = clean_std
    clean_ref.var = clean_var
    clean_ref.sample = lambda: clean_mean + clean_std * torch.randn_like(clean_mean)

    # ---- Init perturbation and mask logits ----
    delta = torch.empty_like(x).uniform_(-eps, eps, generator=generator)
    delta = torch.clamp(delta, -x, 1 - x)
    mask_logits = torch.randn(1, 1, h, w, device=device, dtype=weight_dtype, generator=generator)

    best_total = -float("inf")
    best_loss_at_best = 0.0
    best_pen_at_best = 0.0
    best_x_adv = x.clone()
    prev_mask_proj = project_mask_topk(torch.sigmoid(mask_logits), k)
    unchanged_count = 0
    n_reinits = 0
    last_penalty_val = 0.0

    inner_pbar = tqdm(
        range(args.max_train_steps),
        desc=f"img {image_idx} (step)",
        leave=False,
    )
    for step in inner_pbar:
        delta.requires_grad_(True)
        mask_logits.requires_grad_(True)

        sig_mask = torch.sigmoid(mask_logits)
        proj_perturb, mask_proj = SparseMasking.apply(delta, sig_mask, k)
        x_adv = torch.clamp(x + proj_perturb, 0.0, 1.0)

        poison_latent = vae.encode(2 * x_adv - 1).latent_dist
        loss = compute_loss(
            args.attack_type, clean_ref, poison_latent, args.lambda_constraint
        )

        # ---- Local clustering penalty ----
        if use_cluster:
            mask_proj_diff = mask_proj.detach() + mask_logits - mask_logits.detach()
            penalty = compute_cluster_penalty(
                mask_proj_diff, sens, cluster_kernel, args.cluster_density_threshold
            )
            last_penalty_val = penalty.detach().item()
        else:
            penalty = None

        # ---- Direction-balanced gradient on mask_logits ----
        if use_cluster:
            g_loss_delta, g_loss_mask = torch.autograd.grad(
                loss, [delta, mask_logits], retain_graph=True
            )
            g_pen_mask = torch.autograd.grad(penalty, mask_logits)[0]
            vn = g_loss_mask.norm() + 1e-10
            pn = g_pen_mask.norm() + 1e-10
            grad_mask = g_loss_mask / vn - args.lambda_cluster * g_pen_mask / pn
            grad_delta = g_loss_delta
            total_val = loss.detach().item() - args.lambda_cluster * last_penalty_val
        else:
            grad_delta, grad_mask = torch.autograd.grad(
                loss, [delta, mask_logits]
            )
            total_val = loss.detach().item()

        loss_val = loss.detach().item()

        # ---- Save best by TOTAL objective (loss - lambda * penalty) ----
        if total_val > best_total:
            best_total = total_val
            best_loss_at_best = loss_val
            best_pen_at_best = last_penalty_val
            best_x_adv = x_adv.detach().clone()

        # ---- Update perturbation (PGD sign step) ----
        delta = delta.detach() + alpha * grad_delta.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -x, 1 - x)

        # ---- Update mask logits ----
        if use_cluster:
            mask_logits = mask_logits.detach() + mask_step * grad_mask
        else:
            gnorm = torch.norm(grad_mask, p=2)
            if gnorm > 1e-10:
                mask_logits = mask_logits.detach() + mask_step * grad_mask / (gnorm + 1e-10)
            else:
                mask_logits = mask_logits.detach()

        # ---- Patience-based reinit ----
        with torch.no_grad():
            cur_mask_proj = project_mask_topk(torch.sigmoid(mask_logits), k)
            changed = (cur_mask_proj - prev_mask_proj).abs().sum().item() > 0
            if not changed:
                unchanged_count += 1
            else:
                unchanged_count = 0
            prev_mask_proj = cur_mask_proj
            if unchanged_count >= args.patience:
                mask_logits = torch.randn(
                    1, 1, h, w, device=device, dtype=weight_dtype, generator=generator
                )
                delta = torch.empty_like(x).uniform_(-eps, eps, generator=generator)
                delta = torch.clamp(delta, -x, 1 - x)
                prev_mask_proj = project_mask_topk(torch.sigmoid(mask_logits), k)
                unchanged_count = 0
                n_reinits += 1

        inner_pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            tot=f"{total_val:.4f}",
            best=f"{best_total:.4f}",
            pen=f"{last_penalty_val:.4f}",
            reinits=n_reinits,
        )
    inner_pbar.close()

    return best_x_adv, best_total, best_loss_at_best, best_pen_at_best, n_reinits, last_penalty_val


# --------------------------------------------------------------------------- #
# Sensitivity-map construction.
# --------------------------------------------------------------------------- #
def _normalize_per_image(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize each sample in a (N, 1, H, W) tensor to [0, 1]."""
    n = t.shape[0]
    flat = t.view(n, -1)
    tmin = flat.min(dim=1, keepdim=True).values
    tmax = flat.max(dim=1, keepdim=True).values
    norm = (flat - tmin) / (tmax - tmin + eps)
    return norm.view_as(t).clamp(0, 1)


def build_sensitivity_map(images_m11: torch.Tensor, args):
    """
    Returns a per-image sensitivity map of shape (N, 1, H, W) in [0, 1],
    with high values on perceptually conspicuous regions, or None if
    saliency is fully disabled.
    """
    method = args.saliency_method
    if method == "none":
        print("[Sensitivity] saliency_method=none; cluster penalty disabled.")
        return None

    if method == "bisenet":
        if not args.bisenet_checkpoint:
            raise ValueError(
                "bisenet_checkpoint is required when saliency_method='bisenet'"
            )
        bisenet_model = load_bisenet(args.bisenet_checkpoint)
        saliency, _fg, _parsing = compute_bisenet_maps(images_m11, bisenet_model)
        del bisenet_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:  # face | spectral
        saliency = get_saliency_map(images_m11, method=method)

    sens = _normalize_per_image(saliency.cpu())
    print(f"[Sensitivity] method={method}; "
          f"mean={sens.mean():.3f}, max={sens.max():.3f}")
    return sens


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    vae.eval()

    dataset = InstanceImageDataset(
        instance_data_root=args.instance_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers
    )

    os.makedirs(args.output_dir, exist_ok=True)

    aux_dir = os.path.join(
        os.path.dirname(os.path.abspath(args.output_dir)),
        os.path.basename(os.path.abspath(args.output_dir)) + "_aux",
    )
    os.makedirs(aux_dir, exist_ok=True)
    vis_dir = os.path.join(aux_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(aux_dir, "sparse_log.txt")

    # Pre-compute sensitivity maps for all images.
    all_imgs01 = torch.stack(
        [dataset[i]["pixel_values"] for i in range(len(dataset))]
    )  # (N, C, H, W) in [0, 1]
    sensitivity_all = build_sensitivity_map(all_imgs01 * 2.0 - 1.0, args)

    # Save per-image sensitivity preview for debugging.
    if sensitivity_all is not None:
        sens_vis_dir = os.path.join(aux_dir, "sensitivity")
        os.makedirs(sens_vis_dir, exist_ok=True)
        for i in range(sensitivity_all.shape[0]):
            s = (sensitivity_all[i, 0].clamp(0, 1) * 255).byte().cpu().numpy()
            Image.fromarray(s, mode="L").save(os.path.join(sens_vis_dir, f"{i}.png"))

    h = w = args.resolution
    k = max(1, int(args.sparsity_ratio * h * w))

    # Build clustering kernel once.
    K = args.cluster_kernel_size
    cluster_kernel = torch.ones(1, 1, K, K, device=device, dtype=weight_dtype) / float(K * K)

    header = (
        f"attack | attack_type={args.attack_type} | seed={args.seed}\n"
        f"eps={args.eps}/255 | alpha(frac)={args.alpha} | beta={args.beta}\n"
        f"sparsity_ratio={args.sparsity_ratio} | L0 budget k={k}/{h*w} pixels\n"
        f"max_train_steps={args.max_train_steps} | patience={args.patience} | "
        f"lambda_constraint={args.lambda_constraint}\n"
        f"saliency_method={args.saliency_method}\n"
        f"cluster: K={K}, density_threshold={args.cluster_density_threshold}, "
        f"lambda_cluster={args.lambda_cluster}\n"
        f"sensitivity_available={sensitivity_all is not None}\n\n"
    )
    print(header.strip())
    log_file = open(log_path, "w")
    log_file.write(header)
    log_file.flush()

    to_image = transforms.ToPILImage()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    summary = []
    pbar = tqdm(loader, desc="Images")
    for batch in pbar:
        idx = int(batch["index"].item())
        x = batch["pixel_values"]

        sens_i = None
        if sensitivity_all is not None:
            sens_i = sensitivity_all[idx:idx + 1]

        x_adv, best_total, best_loss, best_pen, n_reinits, last_pen = attack_one_image(
            vae, x, args, k, device, weight_dtype, generator,
            sensitivity_map_img=sens_i,
            cluster_kernel=cluster_kernel,
            image_idx=idx,
        )

        img = to_image(x_adv.squeeze(0).cpu().clamp(0, 1))
        img.save(os.path.join(args.output_dir, f"{idx}.png"))

        # Heatmap + per-image stats.
        x_cpu = x.to(x_adv.device).float()
        pert = (x_adv - x_cpu).abs().squeeze(0)
        pert_pixel = pert.max(dim=0).values  # (H, W)
        nnz = int((pert_pixel > 1e-6).sum().item())
        total = pert_pixel.numel()
        l1 = float(pert.sum().item())
        avg_pert = l1 / max(nnz * pert.shape[0], 1)

        if pert_pixel.max() > 0:
            heat = (pert_pixel / pert_pixel.max() * 255.0).clamp(0, 255).byte().cpu().numpy()
        else:
            heat = pert_pixel.byte().cpu().numpy()
        Image.fromarray(heat, mode="L").save(os.path.join(vis_dir, f"{idx}_heatmap.png"))

        # Cluster stats on the FINAL support.
        cluster_stat = ""
        if sens_i is not None:
            with torch.no_grad():
                support = (pert_pixel > 1e-6).float().unsqueeze(0).unsqueeze(0).to(device)
                density = F.conv2d(support, cluster_kernel, padding=K // 2)
                sens_dev = sens_i.to(device)
                cis = density * sens_dev
                over = F.relu(cis - args.cluster_density_threshold)
                # mean sensitivity on chosen support
                mean_s_on_support = (
                    (sens_dev * support).sum() / support.sum().clamp_min(1)
                ).item()
                cluster_stat = (
                    f" final_penalty={over.mean().item():.5f} "
                    f"frac_over_thr={(over > 0).float().mean().item():.4f} "
                    f"mean_sens_support={mean_s_on_support:.3f}"
                )

        summary.append((idx, best_total, best_loss, best_pen, nnz, nnz / total, avg_pert, n_reinits, last_pen))
        log_file.write(
            f"[img {idx}] best_total={best_total:.4f} best_loss@best={best_loss:.4f} "
            f"best_pen@best={best_pen:.5f} last_pen={last_pen:.5f} "
            f"nnz={nnz}/{total} ({nnz/total*100:.3f}%) avg_pert={avg_pert:.5f} "
            f"reinits={n_reinits}{cluster_stat}\n"
        )
        log_file.flush()

        mem_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
        pbar.set_postfix(
            best_total=f"{best_total:.4f}",
            best_loss=f"{best_loss:.4f}",
            best_pen=f"{best_pen:.4f}",
            nnz=nnz,
            mem_MB=f"{mem_alloc:.0f}",
        )

    if summary:
        avg_total = float(np.mean([s[1] for s in summary]))
        avg_loss = float(np.mean([s[2] for s in summary]))
        avg_pen_best = float(np.mean([s[3] for s in summary]))
        avg_nnz = float(np.mean([s[4] for s in summary]))
        avg_sparsity = float(np.mean([s[5] for s in summary]))
        avg_pert = float(np.mean([s[6] for s in summary]))
        avg_pen_last = float(np.mean([s[8] for s in summary]))
        log_file.write(
            f"\nSummary: avg_best_total={avg_total:.4f}, "
            f"avg_best_loss={avg_loss:.4f}, avg_best_pen={avg_pen_best:.5f}, "
            f"avg_last_pen={avg_pen_last:.5f}, avg_nnz={avg_nnz:.1f}, "
            f"avg_sparsity={avg_sparsity*100:.3f}%, avg_pert={avg_pert:.5f}\n"
        )
    log_file.close()
    print(f"Done. Adv images at {args.output_dir}; heatmaps + log at {aux_dir}")


if __name__ == "__main__":
    main()
