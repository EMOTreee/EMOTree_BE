from datetime import timedelta
from jose import jwt, JWTError
from app.core.config import settings
from app.core.time import now_kst

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = now_kst() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, settings.JWT_ALGORITHM)

def verify_access_token(token: str):
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None
