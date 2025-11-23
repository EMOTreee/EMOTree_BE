from typing import Annotated
from fastapi import Depends
from collections.abc import Generator
from sqlmodel import Session
from app.db.session import engine
from fastapi import Cookie, HTTPException
from app.crud.user import get_user_by_id
from app.models.user import User
from app.utils.jwt_provider import verify_access_token

def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_db)]


