
from datetime import datetime
from sqlmodel import Session, select
from app.models.user import User


def get_user_by_kakao_id(*, session: Session, kakao_id: int) -> User | None:
    statement = select(User).where(User.kakao_id == kakao_id)
    return session.exec(statement).first()
def get_user_by_id(*, session: Session, user_id: int) -> User | None:
    statement = select(User).where(User.id == user_id)
    return session.exec(statement).first()

def upsert_user(
    *,
    session: Session,
    kakao_id: int,
    email: str,
    nickname: str,
) -> User:

    user = get_user_by_kakao_id(session=session, kakao_id=kakao_id)

    if user:
        # UPDATE
        user.email = email
        user.nickname = nickname
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    # INSERT
    new_user = User(
        kakao_id=kakao_id,
        email=email,
        nickname=nickname,
        created_at=datetime.utcnow(),
    )

    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user