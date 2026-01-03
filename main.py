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

def clean_output(text: str) -> str:
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def limit_sentences(text: str, n: int) -> str:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:n])



#request model
class GenerateRequest(BaseModel):
    prompt:  str
    sentences: int=3

@app.get("/health")
def health_check():
    return {"status" : "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail= str(e))