# app/api/feedback.py

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.feedback import Feedback
from app.models.outfit import Outfit, OutfitItem
from app.models.user import User
from app.models.wardrobe import WardrobeItem
from app.services.dev_event_log import log_user_event

router = APIRouter(prefix="/feedback", tags=["Feedback"])

_ACTION_ALIASES: dict[str, str] = {
    "like": "liked",
    "liked": "liked",
    "dislike": "disliked",
    "disliked": "disliked",
    "skip": "skipped",
    "skipped": "skipped",
    "wear": "worn",
    "wore": "worn",
    "worn": "worn",
}

_PREFERENCE_ACTIONS = {"liked", "disliked", "skipped"}


class FeedbackCreate(BaseModel):
    outfit_id: UUID
    action: str


@router.post("")
def create_feedback(
    payload: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    action_raw = payload.action.strip().lower()
    action = _ACTION_ALIASES.get(action_raw)
    if not action:
        raise HTTPException(
            status_code=422,
            detail='Invalid action. Use one of: "liked", "disliked", "worn", "skipped".',
        )

    outfit = (
        db.query(Outfit)
        .filter(
            Outfit.id == payload.outfit_id,
            Outfit.user_id == current_user.id,
        )
        .first()
    )
    if not outfit:
        raise HTTPException(status_code=404, detail="Outfit not found")

    now = datetime.now(timezone.utc)

    if action == "worn":
        existing = (
            db.query(Feedback)
            .filter(
                Feedback.user_id == current_user.id,
                Feedback.outfit_id == outfit.id,
                Feedback.feedback_type == "worn",
            )
            .first()
        )
        if existing:
            log_user_event(
                user_id=current_user.id,
                event="feedback_worn_duplicate",
                meta={"outfit_id": str(outfit.id)},
            )
            return {
                "created": False,
                "feedback_type": existing.feedback_type,
                "feedback_id": existing.id,
            }

        feedback = Feedback(
            user_id=current_user.id,
            outfit_id=outfit.id,
            feedback_type="worn",
        )
        db.add(feedback)

        outfit_item_ids = (
            db.query(OutfitItem.wardrobe_item_id)
            .filter(OutfitItem.outfit_id == outfit.id)
            .all()
        )
        wardrobe_item_ids = [row[0] for row in outfit_item_ids]
        if wardrobe_item_ids:
            (
                db.query(WardrobeItem)
                .filter(
                    WardrobeItem.user_id == current_user.id,
                    WardrobeItem.id.in_(wardrobe_item_ids),
                )
                .update(
                    {
                        WardrobeItem.wear_count: WardrobeItem.wear_count + 1,
                        WardrobeItem.last_worn_at: now,
                    },
                    synchronize_session=False,
                )
            )

        db.commit()
        db.refresh(feedback)
        log_user_event(
            user_id=current_user.id,
            event="feedback_worn",
            meta={"outfit_id": str(outfit.id), "updated_items": len(wardrobe_item_ids)},
        )
        return {
            "created": True,
            "feedback_type": feedback.feedback_type,
            "feedback_id": feedback.id,
            "updated_items": len(wardrobe_item_ids),
        }

    # liked | disliked | skipped (one "preference" record per outfit)
    existing = (
        db.query(Feedback)
        .filter(
            Feedback.user_id == current_user.id,
            Feedback.outfit_id == outfit.id,
            Feedback.feedback_type.in_(sorted(_PREFERENCE_ACTIONS)),
        )
        .order_by(Feedback.created_at.desc())
        .first()
    )
    if existing:
        existing.feedback_type = action
        existing.created_at = now
        db.add(existing)
        db.commit()
        log_user_event(
            user_id=current_user.id,
            event="feedback_preference_updated",
            meta={"outfit_id": str(outfit.id), "action": action},
        )
        return {
            "created": False,
            "feedback_type": existing.feedback_type,
            "feedback_id": existing.id,
        }

    feedback = Feedback(
        user_id=current_user.id,
        outfit_id=outfit.id,
        feedback_type=action,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    log_user_event(
        user_id=current_user.id,
        event="feedback_preference_created",
        meta={"outfit_id": str(outfit.id), "action": action},
    )
    return {
        "created": True,
        "feedback_type": feedback.feedback_type,
        "feedback_id": feedback.id,
    }
