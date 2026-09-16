
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ================================================================
#  Detector-based saliency (no extra checkpoint required)
# ================================================================

def compute_face_saliency(images: torch.Tensor) -> torch.Tensor:
    """
    Haar-cascade face/eye attention map.
    T-zone (eyes-nose-mouth) = high attention, hair/background = low.

    Args:    images (B, 3, H, W) in [-1, 1]
    Returns: (B, 1, H, W) in [0, 1], larger = more visually attended
    """
    B, C, H, W = images.shape
    face_cas = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cas = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    maps = []
    for i in range(B):
        img_np = _to_uint8(images[i])
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        smap = np.zeros((H, W), dtype=np.float32)

        faces = face_cas.detectMultiScale(gray, 1.1, 4, minSize=(H // 8, W // 8))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # whole face: medium attention
                cv2.ellipse(smap, (x + w // 2, y + h // 2),
                            (w // 2, h // 2), 0, 0, 360, 0.5, -1)
                # eyes: highest attention
                roi = gray[y: y + h // 2, x: x + w]
                eyes = eye_cas.detectMultiScale(roi, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.circle(smap, (x + ex + ew // 2, y + ey + eh // 2),
                               max(ew, eh), 1.0, -1)
                # mouth: high attention
                cv2.ellipse(smap, (x + w // 2, y + int(h * 0.78)),
                            (w // 4, h // 10), 0, 0, 360, 0.8, -1)
        else:
            # fallback: centered Gaussian
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            sig = min(H, W) / 4.0
            smap = np.exp(-((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * sig ** 2))

        smap = _smooth_and_norm(smap, H, W)
        maps.append(torch.from_numpy(smap).unsqueeze(0))
    return torch.stack(maps).to(images.device)


def compute_spectral_saliency(images: torch.Tensor) -> torch.Tensor:
    """
    Spectral Residual saliency (Hou & Zhang, CVPR 2007). Not face-specific.

    Args:    images (B, 3, H, W) in [-1, 1]
    Returns: (B, 1, H, W) in [0, 1]
    """
    B, C, H, W = images.shape
    sr = cv2.saliency.StaticSaliencySpectralResidual_create()
    maps = []
    for i in range(B):
        img_bgr = cv2.cvtColor(_to_uint8(images[i]), cv2.COLOR_RGB2BGR)
        ok, smap = sr.computeSaliency(img_bgr)
        if not ok or smap is None:
            smap = np.ones((H, W), dtype=np.float32) * 0.5
        else:
            smap = smap.astype(np.float32)
        smap = _smooth_and_norm(smap, H, W)
        maps.append(torch.from_numpy(smap).unsqueeze(0))
    return torch.stack(maps).to(images.device)


def get_saliency_map(images: torch.Tensor, method: str = "face") -> torch.Tensor:
    """Dispatch for the detector-based back-ends. method: face | spectral | none"""
    if method == "face":
        return compute_face_saliency(images)
    elif method == "spectral":
        return compute_spectral_saliency(images)
    elif method == "none":
        return torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3],
                           device=images.device)
    else:
        raise ValueError(f"Unknown saliency method: {method}")


# ================================================================
#  BiSeNet face parsing (default back-end)
# ================================================================

# 19 face-parsing labels -> human attention weight
FACE_PART_ATTENTION = {
    0: 0.0,    # background       
    1: 0.8,    # skin (cheek/forehead)
    2: 0.7,    # left eyebrow
    3: 0.7,    # right eyebrow
    4: 1.0,    # left eye         
    5: 1.0,    # right eye
    6: 0.6,    # glasses
    7: 0.1,    # left ear
    8: 0.1,    # right ear
    9: 0.1,    # earring
    10: 0.9,   # nose
    11: 0.9,   # mouth
    12: 0.9,   # upper lip
    13: 0.9,   # lower lip
    14: 0.5,   # neck
    15: 0.1,   # necklace
    16: 0.05,  # cloth
    17: 0.02,  # hair             
    18: 0.1,   # hat
}


def load_bisenet(checkpoint_path: str, device: str = "cuda"):
    """
    Load the BiSeNet face-parsing model.

    Args:
        checkpoint_path: pretrained weights (e.g. model/79999_iter.pth)
        device: torch device string
    Returns:
        BiSeNet in eval mode
    """
    from bisenet_model import BiSeNet
    model = BiSeNet(n_classes=19)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"[BiSeNet] Loaded from {checkpoint_path}")
    return model


def compute_bisenet_parsing(images: torch.Tensor, bisenet_model) -> torch.Tensor:
    """
    Run BiSeNet to get per-pixel semantic labels.

    Args:    images (B, 3, H, W) in [-1, 1]
    Returns: parsing (B, H, W) int64 in [0, 18]
    """
    B, C, H, W = images.shape
    device = next(bisenet_model.parameters()).device
    # preprocess: [-1, 1] -> ImageNet-normalized 512x512
    imgs_01 = (images.detach() + 1.0) / 2.0
    imgs_resized = F.interpolate(imgs_01, size=(512, 512),
                                 mode='bilinear', align_corners=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    inp = (imgs_resized - mean) / std

    with torch.no_grad():
        out = bisenet_model(inp.to(device))[0]  # (B, 19, 512, 512)
    if H != 512 or W != 512:
        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
    return out.argmax(1)  # (B, H, W)


def compute_bisenet_maps(images: torch.Tensor, bisenet_model):
    """
    Turn a BiSeNet parsing into a smooth attention map.

    Args:    images (B, 3, H, W) in [-1, 1]
    Returns:
        saliency: (B, 1, H, W) in [0, 1], larger = more visually attended
        fg:       always None (foreground mask is unused by SEAL)
        parsing:  (B, H, W) int64 semantic labels
    """
    parsing = compute_bisenet_parsing(images, bisenet_model)  # (B, H, W)
    B, H, W = parsing.shape

    # per-part attention weight
    smap = torch.zeros(B, H, W, dtype=torch.float32, device=images.device)
    for label, attn in FACE_PART_ATTENTION.items():
        smap[parsing == label] = attn

    # smooth the part boundaries, then min-max normalize
    ksize = max(H, W) // 16
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 3)
    saliency_list = []
    for i in range(B):
        s = smap[i].cpu().numpy()
        s = cv2.GaussianBlur(s, (ksize, ksize), 0)
        lo, hi = s.min(), s.max()
        if hi - lo > 1e-8:
            s = (s - lo) / (hi - lo)
        saliency_list.append(torch.from_numpy(s).unsqueeze(0))
    saliency = torch.stack(saliency_list).to(images.device)  # (B, 1, H, W)

    print(f"[BiSeNet] Parsing done: {B} images, "
          f"saliency range=[{saliency.min():.2f}, {saliency.max():.2f}]")
    return saliency, None, parsing


# ================================================================
#  Helpers
# ================================================================

def _to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [-1, 1] -> (H, W, 3) uint8 RGB"""
    arr = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def _smooth_and_norm(smap: np.ndarray, H: int, W: int) -> np.ndarray:
    """Gaussian smoothing + min-max normalization to [0, 1]."""
    ksize = max(H, W) // 8
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 3)
    smap = cv2.GaussianBlur(smap, (ksize, ksize), 0)
    lo, hi = smap.min(), smap.max()
    if hi - lo > 1e-8:
        smap = (smap - lo) / (hi - lo)
    else:
        smap = np.ones_like(smap) * 0.5
    return smap.astype(np.float32)
