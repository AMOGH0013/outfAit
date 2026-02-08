# app/services/embedding_service.py

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

_IMPORT_ERROR: Exception | None = None
_LOAD_ERROR: Exception | None = None

_MODEL = None
_PREPROCESS = None
_DEVICE = "cpu"

try:
    import torch
    import clip  # type: ignore
except Exception as e:  # pragma: no cover
    _IMPORT_ERROR = e
else:
    try:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL, _PREPROCESS = clip.load("ViT-B/32", device=_DEVICE)
        _MODEL.eval()
    except Exception as e:  # pragma: no cover
        _LOAD_ERROR = e


def _apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Apply a binary-ish mask: keep clothing, black-out background.

    `mask` is expected to be a grayscale image where clothing is white-ish.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    mask_l = mask.convert("L")
    if mask_l.size != image.size:
        mask_l = mask_l.resize(image.size, resample=Image.Resampling.NEAREST)

    # Use mask as alpha to composite onto black background
    black = Image.new("RGB", image.size, (0, 0, 0))
    return Image.composite(image, black, mask_l)


def compute_image_embedding(image_path: str, mask_path: str | None = None) -> list[float]:
    """
    Compute an L2-normalized CLIP ViT-B/32 image embedding.

    - Uses masked image if `mask_path` is provided
    - Returns a plain Python list[float] (no tensors / numpy)
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError("CLIP dependencies not available.") from _IMPORT_ERROR
    if _LOAD_ERROR is not None:
        raise RuntimeError("CLIP model failed to load.") from _LOAD_ERROR
    if _MODEL is None or _PREPROCESS is None:
        raise RuntimeError("CLIP model is not initialized.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if mask_path and not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    try:
        with Image.open(image_path) as im:
            image = im.convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to read image: {image_path}") from e

    if mask_path:
        try:
            with Image.open(mask_path) as m:
                image = _apply_mask(image, m)
        except Exception as e:
            raise ValueError(f"Unable to read mask: {mask_path}") from e

    import torch  # local import for type checkers

    with torch.no_grad():
        image_tensor = _PREPROCESS(image).unsqueeze(0).to(_DEVICE)
        features = _MODEL.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    vec = features[0].detach().float().cpu().tolist()
    return [float(x) for x in vec]


def get_clip_components():
    """
    Return (model, preprocess, device) for reuse by other services.
    Loads at module import time; raises if unavailable.
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError("CLIP dependencies not available.") from _IMPORT_ERROR
    if _LOAD_ERROR is not None:
        raise RuntimeError("CLIP model failed to load.") from _LOAD_ERROR
    if _MODEL is None or _PREPROCESS is None:
        raise RuntimeError("CLIP model is not initialized.")
    return _MODEL, _PREPROCESS, _DEVICE


def _clip_device() -> str:
    """For diagnostics/logging."""
    return _DEVICE
