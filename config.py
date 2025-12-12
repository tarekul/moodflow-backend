from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MAILJET_API_KEY: str
    MAILJET_SECRET_KEY: str
    MAILJET_SENDER_EMAIL: str
    FRONTEND_URL: str = "http://localhost:5174"

    class Config:
        env_file = ".env"

settings = Settings()