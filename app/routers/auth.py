from fastapi import APIRouter, HTTPException, Cookie
from fastapi.responses import RedirectResponse, JSONResponse
from app.config import settings
from app.services.kakao_service import get_token, get_profile
from app.models.user import upsert_user
from app.utils.cookies import clear_token_cookie

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/kakao/login")
def get_kakao_login_url():
    url = (
        "https://kauth.kakao.com/oauth/authorize"
        f"?client_id={settings.KAKAO_CLIENT_ID}"
        f"&redirect_uri={settings.KAKAO_REDIRECT_URI}"
        f"&response_type=code"
    )
    return RedirectResponse(url)

@router.get("/kakao/login2")
def kakao_login(code: str):

    token_res = get_token(code)

    if "access_token" not in token_res:
        raise HTTPException(400, "카카오 access_token 발급 실패")

    access_token = token_res["access_token"]

    profile = get_profile(access_token)

    if "id" not in profile:
        raise HTTPException(400, "카카오 사용자 정보 조회 실패")

    kakao_id = profile["id"]
    nickname = profile["properties"]["nickname"]
    email = profile["kakao_account"].get("email")

    upsert_user(kakao_id, email, nickname)

    redirect_url = "http://localhost:5173"  # 프론트가 받을 페이지

    response = RedirectResponse(url=redirect_url, status_code=302)

    response.set_cookie(
        key="kakao_token",
        value=access_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60*60*6
    )

    return response


@router.get("/user")
def get_user(kakao_token: str = Cookie(None)):

    if not kakao_token:
        raise HTTPException(401, "로그인 필요")

    profile = get_profile(kakao_token)

    if "id" not in profile:
        raise HTTPException(401, "토큰 만료 또는 유효하지 않음")

    nickname = profile["properties"]["nickname"]
    email = profile["kakao_account"].get("email")

    return {"nickname": nickname, "email": email}


@router.get("/logout")
def logout():
    res = JSONResponse({"success": True})
    return clear_token_cookie(res)
