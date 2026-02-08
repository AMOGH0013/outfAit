from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.style_profile import compute_user_style_profile


router = APIRouter(prefix="/style-profile", tags=["StyleProfile"])


@router.get("")
def get_style_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = compute_user_style_profile(db, current_user.id)
    return profile.to_public()

