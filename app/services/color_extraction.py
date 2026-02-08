# app/services/color_extraction.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class ColorExtractionResult:
    primary_color: str
    palette: list[str]
    primary_confidence: float
    palette_confidences: list[float]


def _resize_max_side_bgr(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image_bgr

    scale = max_side / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    if cv2 is not None:
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pillow fallback
    rgb = image_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb, mode="RGB")
    pil = pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    resized_rgb = np.asarray(pil)
    return resized_rgb[:, :, ::-1]


def _load_image_bgr(path: str, max_side: int) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to read image at path: {path}")
        return _resize_max_side_bgr(img, max_side=max_side)

    with Image.open(path) as im:
        im = im.convert("RGB")
        # Resize with Pillow for speed/consistency
        width, height = im.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            im = im.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        rgb = np.asarray(im)
        return rgb[:, :, ::-1]  # RGB -> BGR


def _load_mask_bool(path: str, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw

    if cv2 is not None:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask at path: {path}")
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask > 0

    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != (width, height):
            im = im.resize((width, height), resample=Image.Resampling.NEAREST)
        mask = np.asarray(im)
        return mask > 0


def _bgr_to_hsv_opencv_scale(image_bgr: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    rgb = image_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb, mode="RGB").convert("HSV")
    hsv_pil = np.asarray(pil)  # H: 0-255, S: 0-255, V: 0-255
    h = np.round(hsv_pil[:, :, 0].astype(np.float32) * 179.0 / 255.0).astype(
        np.uint8
    )
    s = hsv_pil[:, :, 1].astype(np.uint8)
    v = hsv_pil[:, :, 2].astype(np.uint8)
    return np.stack([h, s, v], axis=2)


def hsv_to_color_name(h: float, s: float, v: float) -> str:
    """
    Rule-based HSV -> human color mapping.

    Expected input ranges (OpenCV HSV):
      - H: 0..179  (represents 0..360 degrees / 2)
      - S: 0..255
      - V: 0..255
    """
    h_i = float(h)
    s_i = float(s)
    v_i = float(v)

    # Neutrals first (low saturation or very dark)
    if v_i <= 40:
        return "black"
    if s_i <= 25:
        if v_i >= 230:
            return "white"
        return "gray"

    # Approximate neutrals that improve outfit pairing
    if 10 <= h_i <= 30 and v_i <= 120:
        return "brown"
    if 10 <= h_i <= 40 and s_i <= 80 and v_i >= 200:
        return "beige"
    if 90 <= h_i <= 130 and v_i <= 110:
        return "navy"

    # Hue ranges (OpenCV hue scale)
    if h_i < 10 or h_i >= 160:
        return "red"
    if h_i < 25:
        return "orange"
    if h_i < 35:
        return "yellow"
    if h_i < 85:
        return "green"
    if h_i < 130:
        return "blue"
    if h_i < 160:
        return "purple"
    return "red"


def extract_dominant_colors(
    image_path: str,
    mask_path: Optional[str] = None,
    *,
    k: int = 5,
    sample_size: int = 5000,
    max_side: int = 256,
    seed: int = 42,
) -> ColorExtractionResult:
    """
    Deterministic dominant color extraction using HSV + KMeans.

    - Loads image (OpenCV if available, else Pillow)
    - Applies optional mask (keep mask>0)
    - Converts to HSV
    - Samples up to `sample_size` pixels (deterministic)
    - KMeans clusters into `k` dominant colors
    - Maps cluster centers to human color names
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if sample_size < 50:
        raise ValueError("sample_size must be >= 50")
    if max_side < 32:
        raise ValueError("max_side must be >= 32")

    image_bgr = _load_image_bgr(image_path, max_side=max_side)
    hsv = _bgr_to_hsv_opencv_scale(image_bgr)
    height, width = hsv.shape[:2]

    mask_bool: Optional[np.ndarray] = None
    if mask_path:
        mask_bool = _load_mask_bool(mask_path, size_hw=(height, width))

    selection = mask_bool if mask_bool is not None else np.ones((height, width), dtype=bool)

    # MVP background trimming if no segmentation mask: try to remove near-white background.
    if mask_bool is None:
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        non_white = ~((s <= 25) & (v >= 235))
        trimmed = selection & non_white
        if int(trimmed.sum()) >= int(selection.size * 0.05):
            selection = trimmed

    pixels = hsv.reshape(-1, 3)[selection.reshape(-1)]
    if pixels.shape[0] < max(200, k):
        # If mask/background trimming removed too much, fall back to all pixels.
        pixels = hsv.reshape(-1, 3)

    rng = np.random.default_rng(seed)
    if pixels.shape[0] > sample_size:
        idx = rng.choice(pixels.shape[0], size=sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    n_samples = int(sample.shape[0])
    k_eff = min(int(k), n_samples)
    if k_eff <= 0:
        return ColorExtractionResult(
            primary_color="unknown",
            palette=[],
            primary_confidence=0.0,
            palette_confidences=[],
        )

    model = KMeans(n_clusters=k_eff, n_init=10, random_state=seed)
    labels = model.fit_predict(sample.astype(np.float32))
    centers = model.cluster_centers_

    counts = np.bincount(labels, minlength=k_eff).astype(np.float32)
    total = float(counts.sum()) if float(counts.sum()) > 0 else 1.0

    # Sort clusters by prevalence
    order = np.argsort(-counts)

    # Map centers to names, aggregate duplicates (e.g. red at both hue extremes)
    aggregated: dict[str, float] = {}
    for cluster_idx in order.tolist():
        frac = float(counts[cluster_idx] / total)
        h, s, v = centers[cluster_idx].tolist()
        name = hsv_to_color_name(h=h, s=s, v=v)
        aggregated[name] = aggregated.get(name, 0.0) + frac

    palette_sorted = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
    palette = [name for name, _ in palette_sorted]
    confidences = [float(frac) for _, frac in palette_sorted]

    primary_color = palette[0] if palette else "unknown"
    primary_confidence = confidences[0] if confidences else 0.0

    return ColorExtractionResult(
        primary_color=primary_color,
        palette=palette,
        primary_confidence=round(primary_confidence, 4),
        palette_confidences=[round(c, 4) for c in confidences],
    )

