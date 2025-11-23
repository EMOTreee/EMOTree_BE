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

def get_current_user(
    session: SessionDep,
    access_token: str = Cookie(None),
    
):
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = verify_access_token(access_token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub")
    user = get_user_by_id(session=session, user_id=user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user

CurrentUserDep = Annotated[User, Depends(get_current_user)]

