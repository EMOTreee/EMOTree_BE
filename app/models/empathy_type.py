from datetime import datetime
from typing import TYPE_CHECKING, Optional
from sqlmodel import Relationship, SQLModel, Field, Column
from sqlalchemy import Enum as SqlEnum

from app.utils.time import now_kst
from app.models.enums import EmpathyCategory
if TYPE_CHECKING:
    from app.models.user import User

class EmpathyType(SQLModel, table=True):
    __tablename__ = "empathy_type"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(nullable=False, foreign_key="users.id")
    empathy_category: EmpathyCategory = Field(
        sa_column=Column(SqlEnum(EmpathyCategory, name="empathy_category"), nullable=False)
    )
    
    created_at: datetime = Field(default_factory=now_kst, nullable=False)
    user: Optional["User"] = Relationship(back_populates="empathy_type")
