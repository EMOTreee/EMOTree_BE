import requests
from app.core.config import settings
import requests
from sqlmodel import Session

from app.core.config import settings
from app.crud.user import get_user_by_kakao_id, upsert_user
from app.models.user import User
from app.utils.jwt_provider import create_access_token

def get_token(code: str):
    token_url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": settings.KAKAO_CLIENT_ID,
        "redirect_uri": settings.KAKAO_REDIRECT_URI,
        "code": code
    }
    return requests.post(token_url, data=data).json()

def get_profile(access_token: str):
    url = "https://kapi.kakao.com/v2/user/me"
    headers = {"Authorization": f"Bearer {access_token}"}
    return requests.get(url, headers=headers).json()

def kakao_login_service(*, session: Session, code: str) -> str:
    token_res = get_token(code)
    access_token = token_res.get("access_token")

    if not access_token:
        raise ValueError("카카오 access_token 발급 실패")

    profile = get_profile(access_token)

    kakao_id = profile.get("id")
    if not kakao_id:
        raise ValueError("카카오 사용자 정보 조회 실패")

    nickname = profile["properties"]["nickname"]
    email = profile["kakao_account"].get("email")

    user = upsert_user(
        session=session,
        kakao_id=kakao_id,
        email=email,
        nickname=nickname,
    )

    return create_access_token({"sub": str(user.id)})


def upsert_user(*, session: Session, kakao_id: int, email: str, nickname: str) -> User:
    user = get_user_by_kakao_id(session=session, kakao_id=kakao_id)

    if user:
        user.email = email
        user.nickname = nickname
        session.add(user)
    else:
        user = User(
            kakao_id=kakao_id,
            email=email,
            nickname=nickname,
        )
        session.add(user)

    session.commit()
    session.refresh(user)
    return user
