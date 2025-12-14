from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlmodel import SQLModel, Field, Column, Relationship
from sqlalchemy import Enum as SqlEnum, Text

from app.utils.time import now_kst
from app.models.enums import EmotionLabel

if TYPE_CHECKING:
    from app.models.user import User


class EmpathyTrainingResult(SQLModel, table=True):
    __tablename__ = "empathy_training_results"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(nullable=False, foreign_key="users.id")

    emotion_label: EmotionLabel = Field(
        sa_column=Column(SqlEnum(EmotionLabel, name="emotion_label"), nullable=False)
    )

    scenario_text: str = Field(nullable=False)
    user_reply: str = Field(nullable=False)
    empathy_score: int = Field(nullable=False)
    feedback: str = Field(sa_column=Column(Text, nullable=False))

    created_at: datetime = Field(default_factory=now_kst, nullable=False)

    # Relationship
    user: Optional["User"] = Relationship(back_populates="empathy_training_results")
