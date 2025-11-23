from fastapi.responses import JSONResponse

def clear_token_cookie(response: JSONResponse):
    response.delete_cookie(key="access_token", path="/")
    return response
