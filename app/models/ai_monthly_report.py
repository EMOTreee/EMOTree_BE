from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import Text

from sqlmodel import SQLModel, Field, Column, Relationship
from app.utils.time import now_kst

if TYPE_CHECKING:
    from app.models.user import User


class AiMonthlyReport(SQLModel, table=True):
    __tablename__ = "ai_monthly_reports"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(nullable=False, foreign_key="users.id")

    label_year: int | None = Field(nullable=False, default=None)
    label_month: int | None = Field(nullable=False, default=None)


    quiz_analysis: str = Field(sa_column=Column(Text, nullable=True))
    empathy_analysis: str = Field(sa_column=Column(Text, nullable=True))
    expression_analysis: str = Field(sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(default_factory=now_kst, nullable=False)

    # Relationship
    user: Optional["User"] = Relationship(back_populates="ai_monthly_reports")