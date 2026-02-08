# app/services/item_type_suggester.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from app.services.embedding_service import get_clip_components


@dataclass(frozen=True)
class ItemTypeSuggestion:
    item_type: str
    confidence: float


# Keep list small + explainable (no ML training; only CLIP text prompts).
_ITEM_TYPE_PROMPTS: list[tuple[str, str]] = [
    ("shirt", "a photo of a shirt"),
    ("tshirt", "a photo of a t-shirt"),
    ("hoodie", "a photo of a hoodie"),
    ("kurta", "a photo of a kurta"),
    ("jeans", "a photo of a pair of jeans"),
    ("trousers", "a photo of trousers"),
    ("chinos", "a photo of chinos pants"),
    ("shorts", "a photo of shorts"),
]

_MODEL, _PREPROCESS, _DEVICE = get_clip_components()

if torch is None:  # pragma: no cover
    raise RuntimeError("torch is required for item_type_suggester")

import clip  # type: ignore  # noqa: E402

with torch.no_grad():
    _TOKENS = clip.tokenize([p for _t, p in _ITEM_TYPE_PROMPTS]).to(_DEVICE)
    _TEXT_FEATS = _MODEL.encode_text(_TOKENS)
    _TEXT_FEATS = _TEXT_FEATS / _TEXT_FEATS.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def suggest_item_types(
    image_embedding: list[float],
    *,
    top_k: int = 3,
    temperature: float = 0.02,
) -> list[dict]:
    """
    Suggest item_type values using CLIP similarity between the image embedding and
    precomputed text embeddings. Returns ranked suggestions with softmax confidence.

    This does NOT write to DB and must not override user-confirmed item_type.
    """
    if not image_embedding:
        return []
    if top_k < 1:
        return []

    with torch.no_grad():
        img = torch.tensor(image_embedding, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        sims = (img @ _TEXT_FEATS.T).squeeze(0)  # cosine similarities
        # Softmax for a human-friendly confidence; temperature controls sharpness.
        t = float(temperature) if float(temperature) > 0 else 0.02
        probs = torch.softmax(sims / t, dim=0)

        k = min(int(top_k), probs.shape[0])
        top_probs, top_idx = torch.topk(probs, k=k)

    suggestions: list[dict] = []
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        item_type = _ITEM_TYPE_PROMPTS[int(idx)][0]
        suggestions.append(
            {
                "item_type": item_type,
                "confidence": round(float(p), 4),
            }
        )

    return suggestions

