from fastapi import FastAPI, HTTPException, Header, Request, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import re
import hashlib
import json
import uuid
from tasks import run_async_inference, run_async_inference_short
import logging
from services import redis_client, client, clean_output, limit_sentences, HF_MODEL_ID
from prometheus_client import Counter, Histogram, generate_latest
from pythonjsonlogger import jsonlogger




load_dotenv()

logger = logging.getLogger("llm_service")
logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(job_id)s"
)
log_handler.setFormatter(log_formatter)

if not logger.handlers:
    logger.addHandler(log_handler)

logging.getLogger().handlers = logger.handlers
logging.getLogger().setLevel(logging.INFO)



HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_BASE_URL = os.getenv("HF_API_BASE_URL")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set")

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 512))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 150))
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", 600))
#rate limiting config
RATE_LIMIT_WINDOW= 60 #seconds
MAX_REQUESTS_PER_WINDOW= 5

REDIS_HOST= os.getenv("REDIS_HOST","redis")
REDIS_PORT= int(os.getenv("REDIS_PORT",6379))

DEDUP_TTL_SECONDS= int(os.getenv("DEDUP_TTL_SECONDS", 120))

REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["path", "method", "status"] )
LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["path"])





def check_rate_limit(client_id: str):
    key= f"rate_limit: {client_id}"
    
    current= redis_client.incr(key)

    if current == 1:
        redis_client.expire(key, RATE_LIMIT_WINDOW)
    
    if current > MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code= 429,
            detail= "Rate limit exceeded. Try again later."
        )


#before calling the llm model, the backend must decide that is this request expensive or exceeding the limits?
def estimate_input_tokens(prompt:str) -> int:
    return len(prompt) // 4

def estimate_output_tokens(sentence_count:int) -> int:
    return sentence_count*25

def check_inference_budget(prompt:str, sentence_count: int):
    input_tokens= estimate_input_tokens(prompt)
    output_tokens= estimate_output_tokens(sentence_count)
    total_tokens= input_tokens + output_tokens

    if input_tokens > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code= 400,
            detail= "Input prompt exceeds allowed token budget"
        )
    if output_tokens > MAX_OUTPUT_TOKENS:
        raise HTTPException(
            status_code= 400,
            detail= "Requested output exceeds allowed token budget"
        )
    if total_tokens > MAX_TOTAL_TOKENS:
        raise HTTPException(
            status_code= 400,
            detail= "Total inference budget exceeded"
        )

def make_request_fingerprints(*, prompt: str, sentences: int, model_id: str) -> str:
    payload= {
        "prompt" : prompt,
        "sentences" : sentences,
        "model_id": model_id
    }

    raw= json.dumps(payload, sort_keys=True)# keys are sorted alphabetically
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()



def enforce_policies(*, client_id: str, prompt: str, sentence_count: int):
        
    if not client_id:
        raise HTTPException(
            status_code= 400,
            detail= "X-Client-Id header is required."
            )
        
    check_rate_limit(client_id)
    check_inference_budget(prompt, sentence_count)
        
app = FastAPI(title = "LLM Inference ML Service")   

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "path": request.url.path
        }
    )
    return response

@app.middleware("http")
async def metrics_middleware(request, call_next):
    with LATENCY.labels(path=request.url.path).time():
        response = await call_next(request)
    REQUESTS.labels(path=request.url.path, method=request.method, status=response.status_code).inc()
    return response

    
#request model
class GenerateRequest(BaseModel):
    prompt:  str
    sentences: int=3

@app.get("/health")
def health_check():
    return {"status" : "ok"}

@app.post("/generate")
def generate(req: GenerateRequest, x_client_id: str = Header(None)):

    try:

        enforce_policies(
            client_id= x_client_id,
            prompt= req.prompt,
            sentence_count= req.sentences
        )

        fingerprint= make_request_fingerprints(
            prompt= req.prompt,
            sentences= req.sentences,
            model_id= HF_MODEL_ID
        )

        cache_key= f"dedup:{fingerprint}"
        cached= redis_client.get(cache_key)
        
        if cached:
            return{
                "sentences": req.sentences,
                "output": cached,
                "cache": "hit"
            }
        system_prompt= (
            f"Answer in exactly {req.sentences} sentences. "
            "Do not include reasoning or explanations. "
        )
        completion = client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=[
                {"role": "system","content": system_prompt},
                {"role":"user","content": req.prompt}
                      ],
            temperature=0.4,
            max_tokens=256
        )

        raw_output= completion.choices[0].message.content
        cleaned= clean_output(raw_output)
        final= limit_sentences(cleaned, req.sentences)

        redis_client.setex(
            cache_key,
            DEDUP_TTL_SECONDS,
            final
        )

        return{
            "sentences": req.sentences,
            "output": final,
            "cache": "miss"
        }
    except HTTPException:

        raise 

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error:{str(e)}"
            )
    
@app.post("/generate/async")
def generate_async(req: GenerateRequest, x_client_id: str = Header(None)):

    enforce_policies(
        client_id= x_client_id,
        prompt= req.prompt,
        sentence_count= req.sentences
    )

    job_id = str(uuid.uuid4())

    redis_client.set(
        f"job:{job_id}:status",
        "pending"
    )

    if req.sentences <=2:
        run_async_inference_short.delay(job_id, req.prompt, req.sentences)
    else:
        run_async_inference.delay(job_id, req.prompt, req.sentences)
        
    
    return{
        "job_id": job_id,
        "status": "pending"
    }

@app.get("/result/{job_id}")
def get_result(job_id: str):
    
    status = redis_client.get(f"job:{job_id}:status")

    if not status:
        raise HTTPException(
            status_code= 404,
            detail= "job not found"
        )
    
    if status == "completed":
        result = redis_client.get(f"job:{job_id}:result")

        return{
            "job_id": job_id,
            "status": status,
            "output": result
        }
    
    if status == "failed":
        error = redis_client.get(f"job:{job_id}:error")
        
        return{
            "job_id": job_id,
            "status": status,
            "error": error
        }
    
    return{
        "job_id": job_id,
        "status": status
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

    





    
