from sqlalchemy import Date, DateTime, ForeignKey, Numeric, Text
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, date, timezone
import uuid

from app.database import Base
from app.db_types import GUID


class Outfit(Base):
    __tablename__ = "outfits"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    outfit_date: Mapped[date] = mapped_column(Date, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class OutfitItem(Base):
    __tablename__ = "outfit_items"

    outfit_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("outfits.id", ondelete="CASCADE"),
        primary_key=True,
    )
    wardrobe_item_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("wardrobe_items.id"),
        primary_key=True,
    )
