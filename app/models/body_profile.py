from sqlalchemy import String, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timezone
import uuid

from app.database import Base
from app.db_types import GUID


class BodyProfile(Base):
    __tablename__ = "body_profiles"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # "slim" | "average" | "athletic" | "broad"
    body_shape: Mapped[str | None] = mapped_column(String, nullable=True)

    # "slim" | "regular" | "loose"
    fit_preference: Mapped[str | None] = mapped_column(String, nullable=True)

    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[int | None] = mapped_column(Integer, nullable=True)

    user_name: Mapped[str | None] = mapped_column(String, nullable=True)
    # "male" | "female" | "other" | "prefer_not_to_say"
    sex: Mapped[str | None] = mapped_column(String, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # "very_fair" | "fair" | "medium" | "olive" | "brown" | "dark"
    skin_tone: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
