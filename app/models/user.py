from typing import List
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import BigInteger, Column
from datetime import datetime
from app.utils.time import now_kst
from app.models.ai_monthly_report import AiMonthlyReport
from app.models.emotion_expression_result import EmotionExpressionResult
from app.models.emotion_quiz_result import EmotionQuizResult
from app.models.empathy_training_result import EmpathyTrainingResult
from app.models.empathy_type import EmpathyType


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)

    kakao_id: int = Field(
        sa_column=Column(BigInteger,
                         nullable=False,
                         unique=True,
                         index=True)
    )

    email: str = Field(nullable=False, max_length=255)
    nickname: str = Field(nullable=False, max_length=50)
    created_at: datetime = Field(default_factory=now_kst, nullable=False)

    quiz_results: List["EmotionQuizResult"] = Relationship(
        back_populates="user"
    )
    empathy_training_results: List["EmpathyTrainingResult"] = Relationship(
        back_populates="user"
    )
    emotion_expression_results: List["EmotionExpressionResult"] = Relationship(
        back_populates="user"
    )
    ai_monthly_reports: List["AiMonthlyReport"] = Relationship(
        back_populates="user"
    )
    empathy_type: List["EmpathyType"] = Relationship(
        back_populates="user"
    )
