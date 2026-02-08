from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timezone
import uuid

from app.database import Base
from app.db_types import GUID

class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    outfit_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("outfits.id", ondelete="CASCADE"),
        nullable=False,
    )

    feedback_type: Mapped[str] = mapped_column(
        String, nullable=False
    )  # liked | disliked | worn | skipped

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
