from fastapi.responses import JSONResponse

def set_token_cookie(response: JSONResponse, token: str):
    response.set_cookie(
        key="kakao_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60*60*6
    )
    return response

def clear_token_cookie(response: JSONResponse):
    response.delete_cookie(key="kakao_token", path="/")
    return response
