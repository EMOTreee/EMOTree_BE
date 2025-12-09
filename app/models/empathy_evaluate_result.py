from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional


class EmpathyEvaluateResult(SQLModel, table=True):
    __tablename__ = "empathy_evaluate_result"

    id: Optional[int] = Field(default=None, primary_key=True)

    # 사용자 ID (nullable)
    # user_id: Optional[int] = Field(default=None, foreign_key="user.id")

    # 생성된 시나리오
    scenario: str = Field(nullable=False)

    # 사용자가 입력한 메시지
    user_message: str = Field(nullable=False)

    # GPT 평가 점수
    score: int = Field(nullable=False)

    # GPT 피드백
    feedback: str = Field(nullable=False)

    # 저장 시간
    # created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
