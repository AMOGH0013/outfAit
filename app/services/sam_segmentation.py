# app/services/sam_segmentation.py

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

_IMPORT_ERROR: Exception | None = None
_LOAD_ERROR: Exception | None = None

_PREDICTOR = None
_DEVICE = "cpu"

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:  # pragma: no cover
    _IMPORT_ERROR = e
else:
    try:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        _REPO_ROOT = Path(__file__).resolve().parents[2]
        _CHECKPOINT = _REPO_ROOT / "sam_models" / "sam_vit_b.pth"
        if not _CHECKPOINT.exists():
            raise FileNotFoundError(f"SAM checkpoint not found at: {_CHECKPOINT}")

        sam = sam_model_registry["vit_b"](checkpoint=str(_CHECKPOINT))
        sam.to(device=_DEVICE)
        _PREDICTOR = SamPredictor(sam)
    except Exception as e:  # pragma: no cover
        _LOAD_ERROR = e


def segment_clothing(image_path: str, output_mask_path: str) -> str:
    """
    Create a binary clothing mask using SAM.

    - Foreground prompt: single point at image center (label=1)
    - Output: 8-bit grayscale PNG where 255 = clothing, 0 = background
    - Returns: output_mask_path
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "segment_anything/torch not available; install dependencies to enable SAM segmentation."
        ) from _IMPORT_ERROR
    if _LOAD_ERROR is not None:
        raise RuntimeError("SAM model failed to load.") from _LOAD_ERROR
    if _PREDICTOR is None:
        raise RuntimeError("SAM predictor is not initialized.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            image = np.asarray(rgb)
    except Exception as e:
        raise ValueError(f"Unable to read image: {image_path}") from e

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions")

    try:
        _PREDICTOR.set_image(image)
        point_coords = np.array([[width // 2, height // 2]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        masks, _, _ = _PREDICTOR.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
    except Exception as e:
        raise RuntimeError("SAM inference failed") from e

    if masks is None or len(masks) < 1:
        raise RuntimeError("SAM returned no masks")

    mask = (masks[0].astype(np.uint8) * 255)

    out_dir = os.path.dirname(output_mask_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        Image.fromarray(mask, mode="L").save(output_mask_path)
    except Exception as e:
        raise RuntimeError(f"Failed to write mask to: {output_mask_path}") from e

    return output_mask_path


def _sam_device() -> str:
    """For diagnostics/logging."""
    return _DEVICE

