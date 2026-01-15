import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST","redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
BACKEND_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"

celery_app = Celery(
    "llm_tasks",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["tasks"]
)
