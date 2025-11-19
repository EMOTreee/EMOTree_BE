import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    KAKAO_CLIENT_ID: str = os.getenv("KAKAO_CLIENT_ID")
    KAKAO_REDIRECT_URI: str = os.getenv("KAKAO_REDIRECT_URI")

settings = Settings()
