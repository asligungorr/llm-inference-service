import os
import redis
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN= os.getenv("HF_TOKEN")
HF_API_BASE_URL= os.getenv("HF_API_BASE_URL")
HF_MODEL_ID= os.getenv("HF_MODEL_ID")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set")

REDIS_HOST= os.getenv("REDIS_HOST", "redis")
REDIS_PORT= int(os.getenv("REDIS_PORT", 6379))

client = OpenAI(
    base_url=HF_API_BASE_URL,
    api_key=HF_TOKEN
)

redis_client= redis.Redis(
    host= REDIS_HOST,
    port= REDIS_PORT,
    decode_responses=True
)

def clean_output(text: str) -> str:
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()

def limit_sentences(text: str, n: int) -> str:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:n])