from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from uuid import UUID

import math

from sqlalchemy.orm import Session

from app.models.feedback import Feedback
from app.models.outfit import OutfitItem
from app.models.wardrobe import WardrobeItem


_ACTION_WEIGHT: dict[str, float] = {
    "liked": 1.0,
    "worn": 0.8,
    "skipped": -0.4,
    "disliked": -1.0,
}
_HALF_LIFE_DAYS = 45.0
_MAX_FEEDBACK_ROWS = 1000
_MAX_OUTFITS = 250


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _decay(ts: datetime, now: datetime) -> float:
    days = max(0.0, (now - _to_utc(ts)).total_seconds() / 86400.0)
    return 0.5 ** (days / _HALF_LIFE_DAYS)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-12:
        return vec
    return [float(x) / norm for x in vec]


def _weighted_centroid(vectors: Iterable[Sequence[float]], weights: Iterable[float]) -> list[float] | None:
    acc: list[float] | None = None
    total = 0.0
    for v, w in zip(vectors, weights):
        if not v:
            continue
        w = float(w)
        if w <= 0.0:
            continue
        if acc is None:
            acc = [0.0 for _ in range(len(v))]
        if len(v) != len(acc):
            continue
        for i, x in enumerate(v):
            acc[i] += float(x) * w
        total += w
    if not acc or total <= 1e-12:
        return None
    acc = [x / total for x in acc]
    return _l2_normalize(acc)


def _top_k(scores: dict[str, float], k: int = 3) -> list[str]:
    items = [(name, float(score)) for name, score in scores.items() if float(score) > 1e-6]
    items.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in items[:k]]


@dataclass(frozen=True)
class UserStyleProfile:
    computed_at: datetime
    based_on_outfits: int
    positive_outfits: int
    negative_outfits: int

    preferred_top_colors: list[str]
    preferred_bottom_colors: list[str]
    preferred_top_types: list[str]
    preferred_bottom_types: list[str]

    liked_centroid: list[float] | None
    disliked_centroid: list[float] | None

    def to_public(self) -> dict[str, Any]:
        return {
            "computed_at": self.computed_at,
            "based_on_outfits": self.based_on_outfits,
            "positive_outfits": self.positive_outfits,
            "negative_outfits": self.negative_outfits,
            "preferred_top_colors": self.preferred_top_colors,
            "preferred_bottom_colors": self.preferred_bottom_colors,
            "preferred_top_types": self.preferred_top_types,
            "preferred_bottom_types": self.preferred_bottom_types,
        }


def compute_user_style_profile(db: Session, user_id: UUID) -> UserStyleProfile:
    """
    Derive a small, explainable "style fingerprint" from feedback + wear history.

    No schema changes; computed on-the-fly per request.
    The centroid vectors are used internally for scoring; do not expose them to clients.
    """
    now = datetime.now(timezone.utc)

    feedback_rows = (
        db.query(Feedback.outfit_id, Feedback.feedback_type, Feedback.created_at)
        .filter(
            Feedback.user_id == user_id,
            Feedback.feedback_type.in_(tuple(_ACTION_WEIGHT.keys())),
        )
        .order_by(Feedback.created_at.desc())
        .limit(_MAX_FEEDBACK_ROWS)
        .all()
    )

    # Keep only the latest record per (outfit_id, feedback_type).
    per_outfit_actions: dict[UUID, dict[str, datetime]] = {}
    for outfit_id, feedback_type, created_at in feedback_rows:
        acts = per_outfit_actions.setdefault(outfit_id, {})
        if feedback_type in acts:
            continue
        acts[feedback_type] = created_at
        if len(per_outfit_actions) >= _MAX_OUTFITS:
            # Because we iterate in descending time order, this bounds work deterministically.
            continue

    outfit_weights: dict[UUID, float] = {}
    pos = 0
    neg = 0
    for outfit_id, acts in per_outfit_actions.items():
        w = 0.0
        for ft, ts in acts.items():
            base = _ACTION_WEIGHT.get(ft)
            if base is None:
                continue
            w += float(base) * _decay(ts, now)
        if abs(w) < 1e-6:
            continue
        outfit_weights[outfit_id] = w
        if w > 0:
            pos += 1
        else:
            neg += 1

    outfit_ids = list(outfit_weights.keys())
    if not outfit_ids:
        return UserStyleProfile(
            computed_at=now,
            based_on_outfits=0,
            positive_outfits=0,
            negative_outfits=0,
            preferred_top_colors=[],
            preferred_bottom_colors=[],
            preferred_top_types=[],
            preferred_bottom_types=[],
            liked_centroid=None,
            disliked_centroid=None,
        )

    rows = (
        db.query(
            OutfitItem.outfit_id,
            WardrobeItem.category,
            WardrobeItem.item_type,
            WardrobeItem.color,
            WardrobeItem.color_palette,
            WardrobeItem.embedding,
        )
        .join(WardrobeItem, WardrobeItem.id == OutfitItem.wardrobe_item_id)
        .filter(
            WardrobeItem.user_id == user_id,
            OutfitItem.outfit_id.in_(outfit_ids),
        )
        .all()
    )

    top_color_scores: dict[str, float] = {}
    bottom_color_scores: dict[str, float] = {}
    top_type_scores: dict[str, float] = {}
    bottom_type_scores: dict[str, float] = {}

    liked_vectors: list[Sequence[float]] = []
    liked_weights: list[float] = []
    disliked_vectors: list[Sequence[float]] = []
    disliked_weights: list[float] = []

    for outfit_id, category, item_type, color, palette, embedding in rows:
        w = float(outfit_weights.get(outfit_id, 0.0))
        if abs(w) < 1e-6:
            continue

        cat = (category or "").strip().lower()
        itype = (item_type or "").strip().lower()
        col = (color or "").strip().lower()
        pal = palette if isinstance(palette, list) else []

        def bump(d: dict[str, float], key: str, amt: float):
            if not key:
                return
            d[key] = float(d.get(key, 0.0)) + float(amt)

        if cat == "top":
            bump(top_type_scores, itype, w)
            bump(top_color_scores, col, w)
            for p in pal[:5]:
                bump(top_color_scores, str(p).strip().lower(), w * 0.25)
        elif cat == "bottom":
            bump(bottom_type_scores, itype, w)
            bump(bottom_color_scores, col, w)
            for p in pal[:5]:
                bump(bottom_color_scores, str(p).strip().lower(), w * 0.25)

        if embedding and isinstance(embedding, list):
            if w > 0:
                liked_vectors.append(embedding)
                liked_weights.append(abs(w))
            else:
                disliked_vectors.append(embedding)
                disliked_weights.append(abs(w))

    return UserStyleProfile(
        computed_at=now,
        based_on_outfits=len(outfit_ids),
        positive_outfits=pos,
        negative_outfits=neg,
        preferred_top_colors=_top_k(top_color_scores, k=3),
        preferred_bottom_colors=_top_k(bottom_color_scores, k=3),
        preferred_top_types=_top_k(top_type_scores, k=3),
        preferred_bottom_types=_top_k(bottom_type_scores, k=3),
        liked_centroid=_weighted_centroid(liked_vectors, liked_weights),
        disliked_centroid=_weighted_centroid(disliked_vectors, disliked_weights),
    )

