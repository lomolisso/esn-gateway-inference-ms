from celery import Celery
from app.config import CELERY_BROKER_URL

celery = Celery(
    "celery_app",
    broker=CELERY_BROKER_URL,
    include=[
        "app.celery_worker.tasks",
    ]
)