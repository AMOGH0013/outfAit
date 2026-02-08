# app/services/outfit_engine.py

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional, Mapping, Sequence
from uuid import UUID
import random
import math

# -------------------------
# CONSTANTS
# -------------------------

NEUTRAL_COLORS = {
    "black", "white", "grey", "gray", "navy", "beige", "brown"
}

NOVELTY_COOLDOWN_DAYS = 7
FIT_BONUS_MAX = 0.05
STYLE_AFFINITY_MAX = 0.05


# -------------------------
# DATA INTERFACES (Duck Typing)
# -------------------------
# These are NOT ORM models.
# They describe what attributes are expected.

class UserLike:
    id: UUID
    body_shape: Optional[str]
    forbidden_items: Optional[List[str]]
    fit_preference: Optional[str]
    # Optional derived, per-user style fingerprint (computed upstream).
    # Expected keys (if present): preferred_* and centroid vectors.
    style_profile: Optional[dict]


class WardrobeItemLike:
    id: UUID
    category: str            # "top" or "bottom"
    item_type: str           # "shirt", "trouser"
    color: Optional[str]
    fit: Optional[str]
    last_worn_at: Optional[datetime]
    wear_count: int
    is_active: bool
    embedding: Optional[List[float]]


# -------------------------
# FILTERING
# -------------------------

def filter_wardrobe_items(
    items: List[WardrobeItemLike],
    forbidden_items: Optional[List[str]]
) -> List[WardrobeItemLike]:
    """Remove inactive and forbidden items."""
    forbidden = set(forbidden_items or [])
    return [
        item for item in items
        if item.is_active and item.item_type not in forbidden
    ]


# -------------------------
# CANDIDATE GENERATION
# -------------------------

def split_by_category(
    items: List[WardrobeItemLike]
) -> Tuple[List[WardrobeItemLike], List[WardrobeItemLike]]:
    tops = [i for i in items if i.category == "top"]
    bottoms = [i for i in items if i.category == "bottom"]
    return tops, bottoms


def generate_outfit_candidates(
    tops: List[WardrobeItemLike],
    bottoms: List[WardrobeItemLike]
) -> List[Tuple[WardrobeItemLike, WardrobeItemLike]]:
    return [(top, bottom) for top in tops for bottom in bottoms]


# -------------------------
# SCORING FUNCTIONS
# -------------------------

def body_shape_score(user: UserLike, top: WardrobeItemLike) -> float:
    if not user.body_shape:
        return 1.0

    shape = user.body_shape.lower()
    fit = (top.fit or "").lower()

    if shape == "triangle":
        return 1.0 if fit in ("structured", "regular") else 0.6
    if shape == "apple":
        return 0.6 if fit == "tight" else 1.0

    return 1.0


def color_harmony_score(
    top: WardrobeItemLike,
    bottom: WardrobeItemLike
) -> float:
    top_color = (top.color or "").lower()
    bottom_color = (bottom.color or "").lower()

    if top_color in NEUTRAL_COLORS and bottom_color in NEUTRAL_COLORS:
        return 0.8
    if top_color in NEUTRAL_COLORS or bottom_color in NEUTRAL_COLORS:
        return 1.0
    return 0.7


def novelty_score(item: WardrobeItemLike) -> float:
    if not item.last_worn_at:
        return 1.0

    now_utc = datetime.now(timezone.utc)
    last_worn = item.last_worn_at
    if last_worn.tzinfo is None:
        last_worn = last_worn.replace(tzinfo=timezone.utc)

    days_since = (now_utc - last_worn).days
    return min(days_since / NOVELTY_COOLDOWN_DAYS, 1.0)


def _normalize_fit(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    aliases = {
        "tight": "slim",
        "skinny": "slim",
        "slim": "slim",
        "regular": "regular",
        "relaxed": "loose",
        "loose": "loose",
        "oversized": "loose",
    }
    return aliases.get(v, v)


def _default_fit_for_item_type(item_type: str) -> Optional[str]:
    t = (item_type or "").strip().lower()
    defaults = {
        "hoodie": "loose",
        "kurta": "regular",
        "shirt": "regular",
        "tshirt": "regular",
        "jeans": "regular",
        "trousers": "regular",
        "chinos": "regular",
        "shorts": "regular",
    }
    return defaults.get(t)


def fit_preference_score(user: UserLike, items: List[WardrobeItemLike]) -> float:
    """
    Small bounded modifier based on user's fit_preference vs item fit (or a safe default by item_type).
    Returns value in [-0.05, +0.05].
    """
    pref = _normalize_fit(getattr(user, "fit_preference", None))
    if not pref:
        return 0.0
    if pref not in {"slim", "regular", "loose"}:
        return 0.0

    def fit_to_level(f: str) -> int:
        return {"slim": -1, "regular": 0, "loose": 1}.get(f, 0)

    pref_level = fit_to_level(pref)
    per_item: List[float] = []
    for item in items:
        item_fit = _normalize_fit(getattr(item, "fit", None)) or _default_fit_for_item_type(item.item_type)
        if not item_fit or item_fit not in {"slim", "regular", "loose"}:
            continue
        diff = abs(pref_level - fit_to_level(item_fit))
        if diff == 0:
            per_item.append(FIT_BONUS_MAX)
        elif diff == 1:
            per_item.append(0.0)
        else:
            per_item.append(-FIT_BONUS_MAX)

    if not per_item:
        return 0.0
    score = sum(per_item) / float(len(per_item))
    return round(max(-FIT_BONUS_MAX, min(score, FIT_BONUS_MAX)), 3)


# -------------------------
# EMBEDDING DIVERSITY
# -------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity in [0, 1] for non-negative CLIP similarities (embeddings are normalized).
    Robust to non-normalized inputs (will normalize via norms).
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        norm_a += float(x) * float(x)
        norm_b += float(y) * float(y)

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom <= 1e-12:
        return 0.0
    sim = dot / denom
    # Numerical guard
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def _similarity_to_effect(max_sim: float) -> float:
    """
    Map max similarity to bounded penalty/bonus (deterministic, rule-based).

    - >= 0.90: -0.30
    - 0.80-0.90: -0.15
    - 0.70-0.80: -0.05
    - < 0.70: +0.05
    """
    if max_sim >= 0.90:
        return -0.30
    if max_sim >= 0.80:
        return -0.15
    if max_sim >= 0.70:
        return -0.05
    return 0.05


def embedding_diversity_score(
    items: List[WardrobeItemLike],
    recent_item_embeddings: Optional[Sequence[Sequence[float]]],
) -> Tuple[float, Dict[str, float]]:
    """
    Compare each candidate item to recent worn-item embeddings and produce a bounded additive score.

    Returns:
      (score, diagnostics) where diagnostics includes per-item max similarity.

    Cold start: if no recent embeddings, returns (0.0, {}).
    Missing embeddings: that item contributes 0.0 and is omitted from diagnostics.
    """
    if not recent_item_embeddings:
        return 0.0, {}

    effects: List[float] = []
    sims_by_item: Dict[str, float] = {}

    for item in items:
        if not item.embedding:
            continue

        max_sim = 0.0
        for prev in recent_item_embeddings:
            sim = cosine_similarity(item.embedding, prev)
            if sim > max_sim:
                max_sim = sim

        sims_by_item[str(item.id)] = round(max_sim, 4)
        effects.append(_similarity_to_effect(max_sim))

    if not effects:
        return 0.0, {}

    # Average keeps total contribution bounded regardless of number of items in outfit.
    score = sum(effects) / float(len(effects))
    score = max(-0.30, min(score, 0.05))
    return round(score, 3), sims_by_item


# -------------------------
# STYLE AFFINITY (Derived)
# -------------------------

def style_affinity_score(
    user: UserLike,
    top: WardrobeItemLike,
    bottom: WardrobeItemLike,
) -> tuple[float, str | None]:
    """
    Small bounded modifier based on a derived style fingerprint.

    Returns (score in [-0.05, +0.05], reason_or_none).
    No ML training; fully deterministic.
    """
    sp = getattr(user, "style_profile", None)
    if not sp or not isinstance(sp, dict):
        return 0.0, None

    score = 0.0
    reasons: list[str] = []

    def has_any(x) -> bool:
        return bool(x) and isinstance(x, (list, tuple, set))

    top_colors = set(c for c in (sp.get("preferred_top_colors") or []) if isinstance(c, str))
    bottom_colors = set(c for c in (sp.get("preferred_bottom_colors") or []) if isinstance(c, str))
    top_types = set(t for t in (sp.get("preferred_top_types") or []) if isinstance(t, str))
    bottom_types = set(t for t in (sp.get("preferred_bottom_types") or []) if isinstance(t, str))

    tc = (top.color or "").strip().lower()
    bc = (bottom.color or "").strip().lower()
    if tc and tc in top_colors:
        score += 0.02
    if bc and bc in bottom_colors:
        score += 0.02

    if has_any(top_types) and (top.item_type or "").strip().lower() in top_types:
        score += 0.01
    if has_any(bottom_types) and (bottom.item_type or "").strip().lower() in bottom_types:
        score += 0.01

    liked_centroid = sp.get("liked_centroid")
    disliked_centroid = sp.get("disliked_centroid")
    if liked_centroid and top.embedding:
        sim = cosine_similarity(top.embedding, liked_centroid)
        if sim >= 0.85:
            score += 0.01
    if liked_centroid and bottom.embedding:
        sim = cosine_similarity(bottom.embedding, liked_centroid)
        if sim >= 0.85:
            score += 0.01

    if disliked_centroid and top.embedding:
        sim = cosine_similarity(top.embedding, disliked_centroid)
        if sim >= 0.85:
            score -= 0.02
    if disliked_centroid and bottom.embedding:
        sim = cosine_similarity(bottom.embedding, disliked_centroid)
        if sim >= 0.85:
            score -= 0.02

    score = max(-STYLE_AFFINITY_MAX, min(score, STYLE_AFFINITY_MAX))
    score = round(score, 3)

    if abs(score) < 0.01:
        return 0.0, None

    if score > 0:
        reasons.append("matches your learned style preferences")
    else:
        reasons.append("may not match your learned style preferences")

    return score, "; ".join(reasons)


# -------------------------
# FINAL SCORING
# -------------------------

def rule_score(
    user: UserLike,
    top: WardrobeItemLike,
    bottom: WardrobeItemLike,
) -> float:
    score = 0.0
    score += 0.40 * body_shape_score(user, top)
    score += 0.30 * color_harmony_score(top, bottom)
    score += 0.15 * novelty_score(top)
    score += 0.15 * novelty_score(bottom)
    return score


def score_outfit(
    user: UserLike,
    top: WardrobeItemLike,
    bottom: WardrobeItemLike,
    pair_bias: float = 0.0,
    embedding_score: float = 0.0,
    fit_score: float = 0.0,
    style_score: float = 0.0,
) -> float:
    score = rule_score(user, top, bottom) + pair_bias + embedding_score + fit_score + style_score
    score = max(0.0, min(score, 1.0))
    return round(score, 2)


# -------------------------
# EXPLANATION DATA
# -------------------------

def build_explanation_data(
    user: UserLike,
    top: WardrobeItemLike,
    bottom: WardrobeItemLike,
    final_score: float,
    rule_score_value: float,
    pair_bias: float = 0.0,
    embedding_score: float = 0.0,
    fit_score: float = 0.0,
    style_score: float = 0.0,
    style_reason: str | None = None,
    similarity_diagnostics: Optional[Dict[str, float]] = None,
) -> Dict:
    data: Dict = {
        "top": top.item_type,
        "bottom": bottom.item_type,
        "top_color": top.color,
        "bottom_color": bottom.color,
        "body_shape": user.body_shape,
        "final_score": round(final_score, 2),
        "rule_score": round(rule_score_value, 3),
        "feedback_bias": round(pair_bias, 3),
        "embedding_diversity_score": round(embedding_score, 3),
        "fit_score": round(fit_score, 3),
        "style_affinity_score": round(style_score, 3),
        "reasoning": {
            "body_shape": "balanced fit for body shape",
            "color": "good color harmony",
            "novelty": "not worn recently"
        }
    }

    if abs(pair_bias) >= 0.01:
        if pair_bias > 0:
            data["reasoning"]["feedback"] = "boosted from your past feedback"
        else:
            data["reasoning"]["feedback"] = "penalized from your past feedback"

    if abs(embedding_score) >= 0.01:
        if embedding_score > 0:
            data["reasoning"]["embedding_diversity"] = "boosted for visual diversity vs recently worn outfits"
        else:
            data["reasoning"]["embedding_diversity"] = "penalized for visual similarity to recently worn outfits"
        if similarity_diagnostics:
            data["max_similarities"] = similarity_diagnostics

    if abs(fit_score) >= 0.01:
        if fit_score > 0:
            data["reasoning"]["fit"] = "matches your fit preference"
        else:
            data["reasoning"]["fit"] = "mismatches your fit preference"

    if abs(style_score) >= 0.01 and style_reason:
        data["reasoning"]["style"] = style_reason

    return data


# -------------------------
# PICK BEST OUTFIT
# -------------------------

def pick_best_outfit(
    user: UserLike,
    wardrobe_items: List[WardrobeItemLike],
    pair_biases: Optional[Mapping[Tuple[UUID, UUID], float]] = None,
    recent_item_embeddings: Optional[Sequence[Sequence[float]]] = None,
) -> Optional[Dict]:
    filtered = filter_wardrobe_items(
        wardrobe_items,
        user.forbidden_items
    )

    tops, bottoms = split_by_category(filtered)
    candidates = generate_outfit_candidates(tops, bottoms)

    if not candidates:
        return None

    scored = []
    for top, bottom in candidates:
        pair_bias = 0.0
        if pair_biases:
            pair_bias = pair_biases.get((top.id, bottom.id), 0.0)

        embedding_score, sim_diag = embedding_diversity_score(
            items=[top, bottom],
            recent_item_embeddings=recent_item_embeddings,
        )

        fit_score = fit_preference_score(user, [top, bottom])
        style_score, style_reason = style_affinity_score(user, top, bottom)
        rule_score_value = rule_score(user, top, bottom)
        score = score_outfit(
            user,
            top,
            bottom,
            pair_bias=pair_bias,
            embedding_score=embedding_score,
            fit_score=fit_score,
            style_score=style_score,
        )
        scored.append(
            (score, rule_score_value, top, bottom, pair_bias, embedding_score, fit_score, style_score, style_reason, sim_diag)
        )

    scored.sort(key=lambda x: x[0], reverse=True)

    (
        best_score,
        best_rule_score,
        best_top,
        best_bottom,
        best_pair_bias,
        best_embedding_score,
        best_fit_score,
        best_style_score,
        best_style_reason,
        best_sim_diag,
    ) = scored[0]

    return {
        "score": best_score,
        "items": [best_top, best_bottom],
        "explanation_data": build_explanation_data(
            user,
            best_top,
            best_bottom,
            final_score=best_score,
            rule_score_value=best_rule_score,
            pair_bias=best_pair_bias,
            embedding_score=best_embedding_score,
            fit_score=best_fit_score,
            style_score=best_style_score,
            style_reason=best_style_reason,
            similarity_diagnostics=best_sim_diag,
        )
    }


# -------------------------
# WEEKLY PLAN
# -------------------------

def generate_weekly_plan(
    user: UserLike,
    wardrobe_items: List[WardrobeItemLike],
    days: int = 7,
    pair_biases: Optional[Mapping[Tuple[UUID, UUID], float]] = None,
    recent_item_embeddings: Optional[Sequence[Sequence[float]]] = None,
) -> List[Dict]:
    plan = []
    used_item_ids = set()

    for _ in range(days):
        available = [
            item for item in wardrobe_items
            if item.id not in used_item_ids
        ]

        result = pick_best_outfit(
            user,
            available,
            pair_biases=pair_biases,
            recent_item_embeddings=recent_item_embeddings,
        )
        if not result:
            break

        for item in result["items"]:
            used_item_ids.add(item.id)

        plan.append(result)

    return plan
