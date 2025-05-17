# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch
import os
import time
import uvicorn

print("Starting app...")

# ─── Environment Config ────────────────────────────────────────────────────────
MODEL = "model"
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"HF_TOKEN: {HF_TOKEN}")

if HF_TOKEN:
  MODEL = snapshot_download(
    repo_id="zfir/TypeScriptMate",
    token=HF_TOKEN
  )
  print(f"Model files: {os.listdir(MODEL)}")
else:
  print("No HF_TOKEN provided, using local model")

# ─── Load model & tokenizer ────────────────────────────────────────────────────
print(f"Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL)
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

@app.get("/")
def index():
    return {
        "status": "ok",
        "model": MODEL,
        "timestamp": time.time()
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL,
        "timestamp": time.time()
    }
