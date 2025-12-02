from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional
from app.models.enums import EmotionLabel


class EmpathyScenarioResult(SQLModel, table=True):
    __tablename__ = "empathy_scenario_result"

    id: Optional[int] = Field(default=None, primary_key=True)

    # 사용자 ID (nullable)
    # user_id: Optional[int] = Field(default=None, foreign_key="user.id")

    # 감정 라벨
    emotion: EmotionLabel = Field(index=True)

    # GPT가 생성한 시나리오 텍스트
    scenario: str = Field(nullable=False)

    # 생성 시간
    # created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
