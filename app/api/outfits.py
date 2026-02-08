# app/api/outfits.py

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

from app.models.user import User
from app.models.body_profile import BodyProfile
from app.models.feedback import Feedback
from app.models.wardrobe import WardrobeItem
from app.models.outfit import Outfit, OutfitItem
from app.services.outfit_engine import generate_weekly_plan

# You already have these in your project
from app.database import get_db
from app.dependencies import get_current_user
from app.services.style_profile import compute_user_style_profile
from app.services.dev_event_log import log_user_event

router = APIRouter(prefix="/outfits", tags=["Outfits"])

_FEEDBACK_WEIGHTS: dict[str, float] = {
    "liked": 0.08,
    "worn": 0.06,
    "skipped": -0.03,
    "disliked": -0.10,
}
_FEEDBACK_HALF_LIFE_DAYS = 30.0
_MAX_PAIR_BIAS = 0.15
_RECENT_EMBEDDING_DAYS = 7
_RECENT_EMBEDDING_LIMIT = 50


def _feedback_decay(created_at: datetime, now: datetime) -> float:
    ts = created_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    days_since = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (days_since / _FEEDBACK_HALF_LIFE_DAYS)


def _build_pair_biases(db: Session, user_id: UUID) -> dict[tuple[UUID, UUID], float]:
    allowed = tuple(_FEEDBACK_WEIGHTS.keys())
    feedback_subq = (
        db.query(
            Feedback.id.label("feedback_id"),
            Feedback.outfit_id.label("outfit_id"),
            Feedback.feedback_type.label("feedback_type"),
            Feedback.created_at.label("created_at"),
        )
        .filter(
            Feedback.user_id == user_id,
            Feedback.feedback_type.in_(allowed),
        )
        .order_by(Feedback.created_at.desc())
        .limit(300)
        .subquery()
    )

    rows = (
        db.query(
            feedback_subq.c.feedback_id,
            feedback_subq.c.feedback_type,
            feedback_subq.c.created_at,
            OutfitItem.wardrobe_item_id,
            WardrobeItem.category,
        )
        .join(OutfitItem, OutfitItem.outfit_id == feedback_subq.c.outfit_id)
        .join(WardrobeItem, WardrobeItem.id == OutfitItem.wardrobe_item_id)
        .all()
    )

    feedback_items: dict[UUID, dict] = {}
    for feedback_id, feedback_type, created_at, wardrobe_item_id, category in rows:
        entry = feedback_items.setdefault(
            feedback_id,
            {
                "feedback_type": feedback_type,
                "created_at": created_at,
                "top_id": None,
                "bottom_id": None,
            },
        )
        if category == "top":
            entry["top_id"] = wardrobe_item_id
        elif category == "bottom":
            entry["bottom_id"] = wardrobe_item_id

    now = datetime.now(timezone.utc)
    pair_biases: dict[tuple[UUID, UUID], float] = {}
    for entry in feedback_items.values():
        top_id = entry["top_id"]
        bottom_id = entry["bottom_id"]
        if not top_id or not bottom_id:
            continue

        weight = _FEEDBACK_WEIGHTS.get(entry["feedback_type"])
        if not weight:
            continue

        bias = weight * _feedback_decay(entry["created_at"], now)
        key = (top_id, bottom_id)
        pair_biases[key] = pair_biases.get(key, 0.0) + bias

    for key, value in list(pair_biases.items()):
        if value > _MAX_PAIR_BIAS:
            pair_biases[key] = _MAX_PAIR_BIAS
        elif value < -_MAX_PAIR_BIAS:
            pair_biases[key] = -_MAX_PAIR_BIAS

    return pair_biases


def _parse_explanation(explanation: str | None):
    if not explanation:
        return None
    try:
        return json.loads(explanation)
    except json.JSONDecodeError:
        return explanation


def _serialize_outfit(db: Session, outfit: Outfit) -> dict:
    item_ids_rows = (
        db.query(OutfitItem.wardrobe_item_id)
        .filter(OutfitItem.outfit_id == outfit.id)
        .all()
    )
    item_ids = [row[0] for row in item_ids_rows]

    score_val = outfit.score
    try:
        score_val = float(score_val) if score_val is not None else None
    except Exception:
        pass

    return {
        "outfit_id": outfit.id,
        "date": outfit.outfit_date,
        "score": score_val,
        "final_score": score_val,
        "item_ids": item_ids,
        "explanation_data": _parse_explanation(outfit.explanation),
    }


def _get_recent_item_embeddings(db: Session, user_id: UUID) -> list[list[float]]:
    """
    Collect embeddings for recently worn items (by WardrobeItem.last_worn_at).
    This is cheap and avoids depending on outfit-history completeness.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_RECENT_EMBEDDING_DAYS)

    items = (
        db.query(WardrobeItem.embedding)
        .filter(
            WardrobeItem.user_id == user_id,
            WardrobeItem.is_active.is_(True),
            WardrobeItem.last_worn_at.is_not(None),
            WardrobeItem.last_worn_at >= cutoff,
            WardrobeItem.embedding.is_not(None),
        )
        .order_by(WardrobeItem.last_worn_at.desc())
        .limit(_RECENT_EMBEDDING_LIMIT)
        .all()
    )

    result: list[list[float]] = []
    for (emb,) in items:
        if isinstance(emb, list) and emb:
            result.append(emb)
    return result


@router.post("/generate")
def generate_outfits(
    start_date: date,
    days: int = 7,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Load wardrobe (eligible items only)
    wardrobe_items = (
        db.query(WardrobeItem)
        .filter(
            WardrobeItem.user_id == current_user.id,
            WardrobeItem.is_active.is_(True),
            WardrobeItem.item_type != "unknown",
        )
        .all()
    )

    if not wardrobe_items:
        raise HTTPException(
            status_code=400,
            detail='No eligible wardrobe items. Set item_type (not "unknown") for at least one top and one bottom.',
        )

    has_top = any(i.category == "top" for i in wardrobe_items)
    has_bottom = any(i.category == "bottom" for i in wardrobe_items)
    if not (has_top and has_bottom):
        raise HTTPException(
            status_code=400,
            detail='Need at least one eligible top and one eligible bottom (active + item_type set).',
        )

    results = []
    created_count = 0

    pair_biases = _build_pair_biases(db, current_user.id)
    recent_embeddings = _get_recent_item_embeddings(db, current_user.id)

    profile = (
        db.query(BodyProfile)
        .filter(BodyProfile.user_id == current_user.id)
        .first()
    )
    style_profile = compute_user_style_profile(db, current_user.id)
    user_ctx = SimpleNamespace(
        id=current_user.id,
        body_shape=current_user.body_shape,
        forbidden_items=current_user.forbidden_items,
        fit_preference=(profile.fit_preference if profile else None),
        style_profile={
            **style_profile.to_public(),
            "liked_centroid": style_profile.liked_centroid,
            "disliked_centroid": style_profile.disliked_centroid,
        },
    )

    # Generate weekly plan
    weekly_plan = generate_weekly_plan(
        user=user_ctx,
        wardrobe_items=wardrobe_items,
        days=days,
        pair_biases=pair_biases,
        recent_item_embeddings=recent_embeddings,
    )

    for i in range(days):
        outfit_date = start_date + timedelta(days=i)

        # Ensure idempotency: one outfit per day
        existing = (
            db.query(Outfit)
            .filter(
                Outfit.user_id == current_user.id,
                Outfit.outfit_date == outfit_date
            )
            .first()
        )

        if existing:
            results.append({**_serialize_outfit(db, existing), "created": False})
            continue

        if i >= len(weekly_plan):
            # Not enough candidate outfits to fill requested range.
            continue

        plan = weekly_plan[i]

        outfit = Outfit(
            user_id=current_user.id,
            outfit_date=outfit_date,
            score=plan["score"],
            explanation=json.dumps(plan["explanation_data"])
        )

        db.add(outfit)
        db.flush()  # get outfit.id

        for item in plan["items"]:
            db.add(
                OutfitItem(
                    outfit_id=outfit.id,
                    wardrobe_item_id=item.id
                )
            )

        created_count += 1
        results.append(
            {
                "outfit_id": outfit.id,
                "date": outfit_date,
                "score": plan["score"],
                "final_score": plan["score"],
                "item_ids": [i.id for i in plan["items"]],
                "explanation_data": plan["explanation_data"],
                "created": True,
            }
        )

    db.commit()

    log_user_event(
        user_id=current_user.id,
        event="outfits_generate",
        meta={
            "start_date": str(start_date),
            "days": days,
            "eligible_items": len(wardrobe_items),
            "generated": created_count,
            "returned": len(results),
        },
    )
    return {
        "generated": created_count,
        "returned": len(results),
        "outfits": results,
    }


@router.get("/{outfit_date}")
def get_outfit_by_date(
    outfit_date: date,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    outfit = (
        db.query(Outfit)
        .filter(
            Outfit.user_id == current_user.id,
            Outfit.outfit_date == outfit_date
        )
        .first()
    )

    if not outfit:
        raise HTTPException(
            status_code=404,
            detail="Outfit not found"
        )

    items = (
        db.query(WardrobeItem)
        .join(
            OutfitItem,
            OutfitItem.wardrobe_item_id == WardrobeItem.id
        )
        .filter(OutfitItem.outfit_id == outfit.id)
        .all()
    )

    explanation = None
    if outfit.explanation:
        try:
            explanation = json.loads(outfit.explanation)
        except json.JSONDecodeError:
            explanation = outfit.explanation

    return {
        "date": outfit.outfit_date,
        "outfit_id": outfit.id,
        "score": outfit.score,
        "final_score": outfit.score,
        "items": [
            {
                "id": item.id,
                "type": item.item_type,
                "color": item.color
            }
            for item in items
        ],
        "explanation": explanation
    }
