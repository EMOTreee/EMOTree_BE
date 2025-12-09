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

#---테스트용---------------------------------------------------------------------------------

# 임시 테스트용 -> 더미유저 생성
# from app.models.user import User  # User 모델 import
# @router.post("/test-create-user")
# def test_create_user(session: SessionDep):
#     # 이미 존재하는 유저 있으면 삭제 (id=1 기준)
#     existing_user = session.query(User).filter(User.id == 1).first()
#     if existing_user:
#         session.delete(existing_user)
#         session.commit()

#     dummy = User(
#         kakao_id="dummy-kakao",
#         nickname="테스트유저",
#         profile_image=None
#     )
#     session.add(dummy)
#     session.commit()
#     session.refresh(dummy)

#     return {"message": "dummy user created", "user": {"id": dummy.id, "nickname": dummy.nickname}}

# # 임시 테스트용 -> 액세스토큰 발급
# from app.utils.jwt_provider import *
# from fastapi import Response
# @router.post("/test-login")
# async def test_login(response: Response):
#     test_token = create_access_token({"user_id": 1})
#     response.set_cookie(key="access_token", value=test_token, httponly=True)
#     return {"message": "테스트 로그인 완료", "access_token": test_token}
