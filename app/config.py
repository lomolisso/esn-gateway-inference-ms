import os
from dotenv import load_dotenv

# Retrieve enviroment variables from .env file
load_dotenv()

SECRET_KEY: str = os.environ.get("SECRET_KEY")
ESN_API_URL = os.environ.get("ESN_API_URL")
USE_TFLITE: bool = os.environ.get("USE_TFLITE", "True") == "True"


# CORS
ORIGINS: list = [
    ESN_API_URL,
]