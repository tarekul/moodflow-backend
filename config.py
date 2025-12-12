from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: Optional[str] = None
    MAILJET_API_KEY: str
    MAILJET_SECRET_KEY: str
    MAILJET_SENDER_EMAIL: str
    FRONTEND_URL: str = "http://localhost:5174"

    class Config:
        env_file = ".env"

settings = Settings()