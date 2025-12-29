from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title = "LLM Inference ML Service")

class GenerateRequest(BaseModel):
    prompt:  str

@app.get("/health")
def health_check():
    return {"status" : "ok"}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    return {
       "response": f"Dummy response for prompt: {request.prompt}"
       }

