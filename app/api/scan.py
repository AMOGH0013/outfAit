# app/api/scan.py

import os
import shutil
from uuid import uuid4, UUID
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.scan import ScanSession
from app.models.wardrobe import WardrobeItem
from app.models.user import User

from app.dependencies import get_current_user
from app.services.color_extraction import extract_dominant_colors
from app.services.sam_segmentation import segment_clothing
from app.services.embedding_service import compute_image_embedding
from app.services.item_type_suggester import suggest_item_types
from app.services.dev_event_log import log_user_event

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/scan", tags=["Scan"])
@router.post("/start")
def start_scan(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    scan = ScanSession(
        user_id=current_user.id,
        status="pending"
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    log_user_event(user_id=current_user.id, event="scan_start", meta={"scan_id": str(scan.id)})
    return {
        "scan_id": scan.id,
        "status": scan.status
    }
@router.post("/upload/{scan_id}")
def upload_scan_image(
    scan_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    scan = (
        db.query(ScanSession)
        .filter(
            ScanSession.id == scan_id,
            ScanSession.user_id == current_user.id
        )
        .first()
    )

    if not scan:
        raise HTTPException(status_code=404, detail="Scan session not found")

    # Save image locally
    original_name = os.path.basename(file.filename)
    filename = f"{uuid4()}_{original_name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    public_url = f"/uploads/{filename}"
    base_name, _ext = os.path.splitext(filename)
    mask_filename = f"{base_name}_mask.png"
    mask_filepath = os.path.join(UPLOAD_DIR, mask_filename)
    mask_public_url = f"/uploads/{mask_filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Update scan session
    scan.image_url = public_url
    scan.status = "completed"

    try:
        segment_clothing(filepath, mask_filepath)
        colors = extract_dominant_colors(filepath, mask_path=mask_filepath)
        embedding = compute_image_embedding(filepath, mask_path=mask_filepath)
        suggestions = suggest_item_types(embedding, top_k=3)
    except Exception as e:
        log_user_event(
            user_id=current_user.id,
            event="scan_upload_failed",
            meta={"scan_id": str(scan.id), "error": str(e)},
        )
        raise HTTPException(
            status_code=400,
            detail=f"Scan processing failed: {e}",
        )

    color_conf = {
        "color": colors.primary_confidence,
        "color_palette": dict(zip(colors.palette, colors.palette_confidences)),
    }

    # MVP: create exactly ONE wardrobe item per scan.
    # User can correct/assign category later via PUT /wardrobe/{item_id}.
    top_suggestion = (suggestions[0] if suggestions else None)
    item = WardrobeItem(
        user_id=current_user.id,
        image_url=public_url,
        mask_url=mask_public_url,
        item_type="unknown",
        category="top",
        color=colors.primary_color,
        color_palette=colors.palette,
        embedding=embedding,
        suggested_item_type=(top_suggestion["item_type"] if top_suggestion else None),
        suggested_item_type_confidence=(top_suggestion["confidence"] if top_suggestion else None),
        confidence_scores=color_conf,
    )

    db.add(item)
    db.flush()
    created_ids = [item.id]
    db.commit()

    log_user_event(
        user_id=current_user.id,
        event="scan_upload_completed",
        meta={
            "scan_id": str(scan.id),
            "wardrobe_item_id": str(item.id),
            "mask_url": mask_public_url,
            "extracted_color": colors.primary_color,
        },
    )
    return {
        "scan_id": scan.id,
        "status": scan.status,
        "wardrobe_item_created": True,
        "wardrobe_item_ids": created_ids,
        "mask_url": mask_public_url,
        "extracted_color": colors.primary_color,
        "color_palette": colors.palette,
        "color_confidence": colors.primary_confidence,
        "item_type_suggestions": suggestions,
    }
