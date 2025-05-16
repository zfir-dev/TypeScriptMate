# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import time
import uvicorn

os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

# ─── Environment Config ────────────────────────────────────────────────────────
MODEL = os.getenv("MODEL_NAME", "model")
HF_TOKEN = os.getenv("HF_TOKEN")

# ─── Load model & tokenizer ────────────────────────────────────────────────────
print(f"Loading {MODEL} model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL, use_auth_token=HF_TOKEN)
model.eval()
print(f"Model {MODEL} loaded.")

# ─── FastAPI Setup ────────────────────────────────────────────────────────────
app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

@app.post("/complete")
def complete(req: CompletionRequest):
    start = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Completed request in {time.time() - start:.2f}s")
    return {"completion": result[len(req.prompt):]}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL,
        "timestamp": time.time()
    }
