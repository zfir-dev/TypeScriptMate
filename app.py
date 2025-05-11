# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time

MODEL_PATH = "model"

# Wait for model files to appear before loading
print("🔁 Waiting for model files in /model...")
while not os.path.exists(os.path.join(MODEL_PATH, "config.json")) or not os.path.exists(os.path.join(MODEL_PATH, "training_args.bin")) or not os.path.exists(os.path.join(MODEL_PATH, "optimizer.pt")) or not os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")) or not os.path.exists(os.path.join(MODEL_PATH, "scheduler.pt")):
    print("⏳ /model/config.json or /model/training_args.bin or /model/optimizer.pt or /model/model.safetensors or /model/scheduler.pt not found. Retrying in 2s...")
    time.sleep(2)

app = FastAPI()

# Load model from host-mounted directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

@app.post("/complete")
def complete(req: CompletionRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=req.max_tokens, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"completion": result[len(req.prompt):]}
