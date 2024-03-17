import os
from dotenv import load_dotenv

# Retrieve enviroment variables from .env file
load_dotenv()

SECRET_KEY: str = os.environ.get("SECRET_KEY")
ESN_API_URL : str = os.environ.get("ESN_API_URL")
USE_TFLITE: bool = os.environ.get("USE_TFLITE", "True") == "True"
IS_GATEWAY: bool = USE_TFLITE

ESN_REDIS_URL : str = os.environ.get("ESN_REDIS_URL")
CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL")

# CORS
ORIGINS: list = [
    ESN_API_URL,
]