from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
from openai import OpenAI
import re
import time
from collections import defaultdict

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

rate_limit_store= defaultdict(list)


client = OpenAI(
    base_url=HF_API_BASE_URL,
    api_key=HF_TOKEN
)

def check_rate_limit(client_id: str):
    now= time.time()

    timestamps= rate_limit_store[client_id]

    rate_limit_store[client_id] = [
        ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW
    ]

    if len(rate_limit_store[client_id]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code= 429,
            detail= "Rate limit exceeded. Try again later."
        )
    rate_limit_store[client_id].append(now)

app = FastAPI(title = "LLM Inference ML Service")

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
    

#request model
class GenerateRequest(BaseModel):
    prompt:  str
    sentences: int=3

@app.get("/health")
def health_check():
    return {"status" : "ok"}

@app.post("/generate")
def generate(req: GenerateRequest, x_client_id: str = Header(None)):
    if not x_client_id:
        raise HTTPException(
            status_code=400,
            detail= "X-Client-Id header is required"
        )
    try:
        check_inference_budget(req.prompt, req.sentences)
        check_rate_limit(x_client_id)

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

        return{
            "sentences": req.sentences,
            "output": final
        }
    except HTTPException:

        raise 

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error:{str(e)}"
            )