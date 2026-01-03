from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
from openai import OpenAI
import re

load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_BASE_URL = os.getenv("HF_API_BASE_URL")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set")

client = OpenAI(
    base_url=HF_API_BASE_URL,
    api_key=HF_TOKEN
)


app = FastAPI(title = "LLM Inference ML Service")


#request model
class GenerateRequest(BaseModel):
    prompt:  str

@app.get("/health")
def health_check():
    return {"status" : "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        completion = client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=[
                {"role":"user","content": req.prompt}
                      ],
            temperature=0.7,
            max_tokens=100
        )
        return{
            "output": completion.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail= str(e))