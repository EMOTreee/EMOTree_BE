import requests
from app.config import settings

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
