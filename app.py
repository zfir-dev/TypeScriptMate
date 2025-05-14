# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
import gradio as gr

MODEL = os.getenv("MODEL_NAME", "model")
HF_TOKEN = os.getenv("HF_TOKEN")

# ─── Load model & tokenizer ────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL, token=HF_TOKEN)
model.eval()
print("Model loaded.")

# ─── FastAPI Setup ────────────────────────────────────────────────────────────
app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

@app.post("/complete")
def complete(req: CompletionRequest):
    import time
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
    """Health check endpoint to verify the API is running"""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "timestamp": time.time()
    }
