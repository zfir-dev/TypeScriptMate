# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import time

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "model"
REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json"
    # ✅ tokenizer.json is excluded on purpose (rebuild from source files)
]

# ─── Wait for model files (except tokenizer.json) ─────────────────────────────
def wait_for_model_files(timeout=60):
    print(f"⏳ Waiting for model files in '{MODEL_PATH}' ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(MODEL_PATH, f))]
        if not missing:
            print("✅ All required model files found.")
            return
        print(f"⏳ Still waiting for: {missing}")
        time.sleep(2)
    raise RuntimeError(f"❌ Timeout: Model files not fully copied in {timeout} seconds.")

wait_for_model_files()

# ─── Load tokenizer (safe on all architectures) ───────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding is defined

# ─── Load GPT-2 base and apply LoRA adapter ───────────────────────────────────
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

# ─── FastAPI Setup ────────────────────────────────────────────────────────────
app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

@app.post("/complete")
def complete(req: CompletionRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"completion": result[len(req.prompt):]}
