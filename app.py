# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time

MODEL_PATH = "model"
REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json"
]

def wait_for_model_files(timeout=60):
    print(f"⏳ Waiting for model files in '{MODEL_PATH}' ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(MODEL_PATH, f))]
        if not missing:
            print("All model files found. Proceeding to load.")
            return
        print(f"Still waiting for: {missing}")
        time.sleep(2)
    raise RuntimeError(f"Timeout: Model files not fully copied in {timeout} seconds.")

# ─── Wait before loading ──────────────────────────────────────────────────────
wait_for_model_files()

# ─── Load model & tokenizer ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

# ─── FastAPI Setup ────────────────────────────────────────────────────────────
app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

@app.post("/complete")
def complete(req: CompletionRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=req.max_tokens, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"completion": result[len(req.prompt):]}
