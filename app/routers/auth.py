from fastapi import APIRouter, HTTPException, Cookie
from fastapi.responses import RedirectResponse, JSONResponse
from app.core.config import settings
from app.schemas.auth_schema import CodeRequest
from app.services.kakao_service import get_token, get_profile
from app.routers.dependencies import CurrentUserDep, SessionDep, get_current_user, get_db as get_session
from app.utils.cookies import clear_token_cookie
from app.services.kakao_service import kakao_login_service


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
def kakao_login(code: str, session: SessionDep):

    try:
        access_token = kakao_login_service(session=session, code=code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    redirect_url = "http://localhost:5173"
    response = RedirectResponse(url=redirect_url, status_code=302)
    

    # secure=True 는 https 환경에서만 가능하니 dev에서는 False
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60*60*6
    )

    return response

@router.get("/logout")
def logout():
    res = JSONResponse({"success": True})
    return clear_token_cookie(res)
