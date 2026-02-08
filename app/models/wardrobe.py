from sqlalchemy import (
    String, DateTime, ForeignKey, Boolean, Integer, JSON, Float
)
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timezone
import uuid

from app.database import Base
from app.db_types import GUID


class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    image_url: Mapped[str] = mapped_column(String, nullable=False)
    mask_url: Mapped[str | None] = mapped_column(String, nullable=True)

    item_type: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)

    color: Mapped[str | None] = mapped_column(String, nullable=True)
    color_palette: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    suggested_item_type: Mapped[str | None] = mapped_column(String, nullable=True)
    suggested_item_type_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    pattern: Mapped[str | None] = mapped_column(String, nullable=True)
    fabric: Mapped[str | None] = mapped_column(String, nullable=True)
    fit: Mapped[str | None] = mapped_column(String, nullable=True)
    size: Mapped[str | None] = mapped_column(String, nullable=True)

    season_tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    brand: Mapped[str | None] = mapped_column(String, nullable=True)

    measurements: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence_scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    wear_count: Mapped[int] = mapped_column(Integer, default=0)
    last_worn_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
