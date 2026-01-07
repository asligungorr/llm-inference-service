from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
from openai import OpenAI
import re
import redis
import hashlib
import json

load_dotenv()


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

client = OpenAI(
    base_url=HF_API_BASE_URL,
    api_key=HF_TOKEN
)

redis_client= redis.Redis(
    host= REDIS_HOST,
    port= REDIS_PORT,
    decode_responses= True
)

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


def clean_output(text: str) -> str:
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def limit_sentences(text: str, n: int) -> str:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:n])

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
            max_tokens=100
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