import os
from typing import List
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", "https://localhost:3000"]

    # S3 Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-2")
    S3_MODELS_BUCKET: str = os.getenv(
        "S3_MODELS_BUCKET", "whos-my-good-boy-models")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    JSON_LOGS: bool = os.getenv("JSON_LOGS", "").lower() == "true"

    ENV: str = os.getenv("env", "development")

    MODEL_SOURCE: str = os.getenv("MODEL_SOURCE", "local")

    MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")

    model_config = ConfigDict(
        extra='allow',
        env_file=".env",
        case_sensitive=True
    )


config = Config()
