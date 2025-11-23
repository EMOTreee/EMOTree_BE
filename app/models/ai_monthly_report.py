from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlmodel import SQLModel, Field, Relationship
from app.core.time import now_kst

if TYPE_CHECKING:
    from app.models.user import User


class AiMonthlyReport(SQLModel, table=True):
    __tablename__ = "ai_monthly_reports"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(nullable=False, foreign_key="users.id")

    label_year: int | None = Field(nullable=False, default=None)
    label_month: int | None = Field(nullable=False, default=None)

    quiz_analysis: str = Field(nullable=False)        
    empathy_analysis: str = Field(nullable=False)     
    expression_analysis: str = Field(nullable=False)  

    created_at: datetime = Field(default_factory=now_kst, nullable=False)

    # Relationship
    user: Optional["User"] = Relationship(back_populates="ai_monthly_reports")