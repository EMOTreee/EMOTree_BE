from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    KAKAO_CLIENT_ID: str
    KAKAO_REDIRECT_URI: str

    OPENAI_API_KEY: str
    OPENAI_ORGANIZATION_ID: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    MYSQL_SERVER: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str
    MYSQL_DB: str = "emotree"

    STATIC_IMAGE_DIR: str = "static/images"

    # SQLAlchemy MySQL Connection URI
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
            "?charset=utf8mb4"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

settings = Settings()
