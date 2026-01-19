import os
from celery import Celery
from dotenv import load_dotenv
from kombu import Queue
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

celery_app.conf.update(
    task_soft_time_limit=25,
    task_time_limit=30,
    result_expires=3600,
)
celery_app.conf.task_queues=(
    Queue("short"),
    Queue("long"),
)
celery_app.conf.task_routes = {
    "tasks.run_async_inference": {"queue":"long"},
    "tasks.run_async_inference_short": {"queue":"short"},
}
