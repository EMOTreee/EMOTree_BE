from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlmodel import SQLModel, Field, Column, Relationship
from sqlalchemy import Enum as SqlEnum, Text

from app.utils.time import now_kst
from app.models.enums import EmotionLabel

if TYPE_CHECKING:
    from app.models.user import User


class EmotionExpressionResult(SQLModel, table=True):
    __tablename__ = "emotion_expression_results"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(nullable=False, foreign_key="users.id")

    target_emotion: EmotionLabel = Field(
        sa_column=Column(SqlEnum(EmotionLabel, name="target_emotion"), nullable=False)
    )

    detected_emotion: EmotionLabel = Field(
        sa_column=Column(SqlEnum(EmotionLabel, name="detected_emotion"), nullable=False)
    )

    expression_score: int = Field(nullable=False)
    feedback: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=now_kst, nullable=False)

    user: Optional["User"] = Relationship(back_populates="emotion_expression_results")
