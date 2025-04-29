# filepath: /Users/andresap/repos/whos-my-good-boy/backend/app/core/config.py

import os
from typing import List
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", "https://localhost:3000"]

    # S3 Configuration (for future use)
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv(
        "S3_BUCKET_NAME", "whos-my-good-boy-models")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    JSON_LOGS: bool = os.getenv("JSON_LOGS", "").lower() == "true"

    ENV: str = os.getenv("env", "development")

    model_config = ConfigDict(
        extra='allow',
        env_file=".env",
        case_sensitive=True
    )


settings = Settings()
