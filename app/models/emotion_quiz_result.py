from datetime import datetime
from typing import TYPE_CHECKING, Optional
from sqlmodel import Relationship, SQLModel, Field, Column
from sqlalchemy import Enum as SqlEnum

from app.core.time import now_kst
from app.models.enums import EmotionLabel
if TYPE_CHECKING:
    from app.models.user import User

class EmotionQuizResult(SQLModel, table=True):
    __tablename__ = "emotion_quiz_results"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(nullable=False, foreign_key="users.id")
    emotion_label: EmotionLabel = Field(
        sa_column=Column(SqlEnum(EmotionLabel, name="emotion_label"), nullable=False)
    )
    is_correct: bool = Field(nullable=False)
    created_at: datetime = Field(default_factory=now_kst, nullable=False)

    user: Optional["User"] = Relationship(back_populates="quiz_results")
