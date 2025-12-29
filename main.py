from fastapi import FastAPI

app = FastAPI(title = "LLM Inference ML Service")

@app.get("/health")
def health_check():
    return {"status" : "ok"}

