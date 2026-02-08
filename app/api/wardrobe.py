# app/api/wardrobe.py

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.wardrobe import WardrobeItem
from app.services.dev_event_log import log_user_event

router = APIRouter(prefix="/wardrobe", tags=["Wardrobe"])

ALLOWED_ITEM_TYPES: set[str] = {
    "unknown",
    # tops
    "shirt",
    "tshirt",
    "kurta",
    "hoodie",
    # bottoms
    "jeans",
    "trousers",
    "chinos",
    "shorts",
}

TOP_ITEM_TYPES: set[str] = {"shirt", "tshirt", "kurta", "hoodie"}
BOTTOM_ITEM_TYPES: set[str] = {"jeans", "trousers", "chinos", "shorts"}


def _to_public_upload_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("/uploads/") or value.startswith("http://") or value.startswith("https://"):
        return value
    filename = os.path.basename(value.replace("\\", "/"))
    if not filename:
        return value
    return f"/uploads/{filename}"


def _wardrobe_item_to_dict(item: WardrobeItem) -> dict[str, Any]:
    embedding_dim = None
    try:
        if item.embedding:
            embedding_dim = len(item.embedding)
    except Exception:
        embedding_dim = None

    return {
        "id": item.id,
        "image_url": item.image_url,
        "image_preview_url": _to_public_upload_url(item.image_url),
        "mask_url": item.mask_url,
        "mask_preview_url": _to_public_upload_url(item.mask_url),
        "item_type": item.item_type,
        "category": item.category,
        "color": item.color,
        "color_palette": item.color_palette,
        "has_embedding": bool(item.embedding),
        "embedding_dim": embedding_dim,
        "suggested_item_type": item.suggested_item_type,
        "suggested_item_type_confidence": item.suggested_item_type_confidence,
        "pattern": item.pattern,
        "fabric": item.fabric,
        "fit": item.fit,
        "size": item.size,
        "season_tags": item.season_tags,
        "brand": item.brand,
        "measurements": item.measurements,
        "confidence_scores": item.confidence_scores,
        "wear_count": item.wear_count,
        "last_worn_at": item.last_worn_at,
        "is_active": item.is_active,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


class WardrobeItemUpdate(BaseModel):
    item_type: str | None = None
    category: str | None = None
    color: str | None = None
    color_palette: list[str] | None = None
    pattern: str | None = None
    fabric: str | None = None
    fit: str | None = None
    size: str | None = None
    season_tags: list[str] | None = None
    brand: str | None = None
    measurements: dict[str, Any] | None = None
    confidence_scores: dict[str, Any] | None = None
    is_active: bool | None = None


@router.get("")
def list_wardrobe(
    include_inactive: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(WardrobeItem).filter(WardrobeItem.user_id == current_user.id)
    if not include_inactive:
        query = query.filter(WardrobeItem.is_active.is_(True))

    items = query.order_by(WardrobeItem.created_at.desc()).all()
    return {"count": len(items), "items": [_wardrobe_item_to_dict(i) for i in items]}


@router.put("/{item_id}")
def update_wardrobe_item(
    item_id: UUID,
    payload: WardrobeItemUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = (
        db.query(WardrobeItem)
        .filter(
            WardrobeItem.id == item_id,
            WardrobeItem.user_id == current_user.id,
        )
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Wardrobe item not found")

    updates = (
        payload.model_dump(exclude_unset=True)
        if hasattr(payload, "model_dump")
        else payload.dict(exclude_unset=True)
    )
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    if "category" in updates and updates["category"] is not None:
        category = updates["category"].strip().lower()
        if category not in {"top", "bottom"}:
            raise HTTPException(
                status_code=422,
                detail='Invalid category. Use "top" or "bottom".',
            )
        updates["category"] = category

    if "item_type" in updates and updates["item_type"] is not None:
        item_type = updates["item_type"].strip().lower()
        if item_type not in ALLOWED_ITEM_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid item_type. Allowed: {sorted(ALLOWED_ITEM_TYPES)}",
            )
        updates["item_type"] = item_type

    # Optional consistency check (prevents obvious bad labels)
    next_category = updates.get("category", item.category)
    next_item_type = updates.get("item_type", item.item_type)
    if next_item_type and next_item_type != "unknown":
        if next_category == "top" and next_item_type in BOTTOM_ITEM_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f'item_type "{next_item_type}" is a bottom; set category="bottom" first.',
            )
        if next_category == "bottom" and next_item_type in TOP_ITEM_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f'item_type "{next_item_type}" is a top; set category="top" first.',
            )

    for field, value in updates.items():
        setattr(item, field, value)

    db.add(item)
    db.commit()
    db.refresh(item)
    log_user_event(
        user_id=current_user.id,
        event="wardrobe_update",
        meta={"item_id": str(item.id), "fields": sorted(list(updates.keys()))},
    )
    return _wardrobe_item_to_dict(item)


@router.delete("/{item_id}")
def delete_wardrobe_item(
    item_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = (
        db.query(WardrobeItem)
        .filter(
            WardrobeItem.id == item_id,
            WardrobeItem.user_id == current_user.id,
        )
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Wardrobe item not found")

    item.is_active = False
    db.add(item)
    db.commit()
    log_user_event(
        user_id=current_user.id,
        event="wardrobe_soft_delete",
        meta={"item_id": str(item.id)},
    )
    return {"deleted": True, "id": item.id}
