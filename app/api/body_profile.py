# app/api/body_profile.py

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.body_profile import BodyProfile
from app.models.user import User
from app.services.dev_event_log import log_user_event

router = APIRouter(prefix="/body-profile", tags=["BodyProfile"])

ALLOWED_BODY_SHAPES = {
    "slim",
    "average",
    "athletic",
    "broad",
    "rectangle",
    "triangle",
    "inverted_triangle",
    "oval",
}
ALLOWED_FIT_PREFERENCES = {"slim", "regular", "loose"}
ALLOWED_SEX = {"male", "female", "other", "prefer_not_to_say"}
ALLOWED_SKIN_TONE = {"very_fair", "fair", "medium", "olive", "brown", "dark"}


class BodyProfileUpdate(BaseModel):
    user_name: str | None = None
    sex: str | None = None
    age: int | None = None
    skin_tone: str | None = None
    body_shape: str | None = None
    fit_preference: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None


def _serialize(profile: BodyProfile) -> dict:
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "user_name": profile.user_name,
        "sex": profile.sex,
        "age": profile.age,
        "skin_tone": profile.skin_tone,
        "body_shape": profile.body_shape,
        "fit_preference": profile.fit_preference,
        "height_cm": profile.height_cm,
        "weight_kg": profile.weight_kg,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


@router.get("")
def get_body_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = (
        db.query(BodyProfile)
        .filter(BodyProfile.user_id == current_user.id)
        .first()
    )
    if not profile:
        profile = BodyProfile(user_id=current_user.id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return _serialize(profile)


@router.put("")
def update_body_profile(
    payload: BodyProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    updates = (
        payload.model_dump(exclude_unset=True)
        if hasattr(payload, "model_dump")
        else payload.dict(exclude_unset=True)
    )
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    profile = (
        db.query(BodyProfile)
        .filter(BodyProfile.user_id == current_user.id)
        .first()
    )
    if not profile:
        profile = BodyProfile(user_id=current_user.id)
        db.add(profile)
        db.flush()

    if "body_shape" in updates and updates["body_shape"] is not None:
        value = updates["body_shape"].strip().lower()
        if value not in ALLOWED_BODY_SHAPES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid body_shape. Allowed: {sorted(ALLOWED_BODY_SHAPES)}",
            )
        profile.body_shape = value

    if "fit_preference" in updates and updates["fit_preference"] is not None:
        value = updates["fit_preference"].strip().lower()
        if value not in ALLOWED_FIT_PREFERENCES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid fit_preference. Allowed: {sorted(ALLOWED_FIT_PREFERENCES)}",
            )
        profile.fit_preference = value

    if "user_name" in updates and updates["user_name"] is not None:
        value = updates["user_name"].strip()
        if len(value) > 80:
            raise HTTPException(status_code=422, detail="user_name must be at most 80 characters")
        profile.user_name = value or None

    if "sex" in updates and updates["sex"] is not None:
        value = updates["sex"].strip().lower()
        if value not in ALLOWED_SEX:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid sex. Allowed: {sorted(ALLOWED_SEX)}",
            )
        profile.sex = value

    if "age" in updates and updates["age"] is not None:
        value = int(updates["age"])
        if value < 10 or value > 100:
            raise HTTPException(status_code=422, detail="age must be between 10 and 100")
        profile.age = value

    if "skin_tone" in updates and updates["skin_tone"] is not None:
        value = updates["skin_tone"].strip().lower()
        if value not in ALLOWED_SKIN_TONE:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid skin_tone. Allowed: {sorted(ALLOWED_SKIN_TONE)}",
            )
        profile.skin_tone = value

    if "height_cm" in updates and updates["height_cm"] is not None:
        value = int(updates["height_cm"])
        if value < 50 or value > 250:
            raise HTTPException(status_code=422, detail="height_cm must be between 50 and 250")
        profile.height_cm = value

    if "weight_kg" in updates and updates["weight_kg"] is not None:
        value = int(updates["weight_kg"])
        if value < 20 or value > 400:
            raise HTTPException(status_code=422, detail="weight_kg must be between 20 and 400")
        profile.weight_kg = value

    db.add(profile)
    db.commit()
    db.refresh(profile)
    log_user_event(
        user_id=current_user.id,
        event="body_profile_update",
        meta={"fields": sorted(list(updates.keys()))},
    )
    return _serialize(profile)
