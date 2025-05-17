# app.py

import os
import time
import threading

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from starlette.concurrency import run_in_threadpool

# ─── Thread & BLAS tuning ─────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
HF_TOKEN = os.getenv("HF_TOKEN")

print("Starting app...")

app = FastAPI()

# globals to be populated by load_model()
MODEL_PATH: str = None  # path or repo snapshot
tokenizer: AutoTokenizer = None
model: torch.nn.Module = None

def load_model():
    global MODEL_PATH, tokenizer, model

    # ─── fetch or local ─────────────────────────────────────────────────────────
    if HF_TOKEN:
        MODEL_PATH = snapshot_download(
            repo_id="zfir/TypeScriptMate",
            token=HF_TOKEN
        )
        print(f"Model files: {os.listdir(MODEL_PATH)}")
    else:
        MODEL_PATH = "model"
        print("No HF_TOKEN provided, using local model directory")

    # ─── load & optimize ────────────────────────────────────────────────────────
    print("Loading tokenizer & model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).eval()

    # dynamic 8-bit quantization
    model_q = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # PyTorch 2.0 compile (JIT + graph optimizations)
    model = torch.compile(model_q)
    model.eval()
    print("Model loaded and optimized.")

    # ─── warm up ─────────────────────────────────────────────────────────────────
    print("Warming up…")
    dummy = tokenizer("Hi", return_tensors="pt")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=1)
    print("Warm-up complete.")


# start background loading so /health and docs come up quickly
threading.Thread(target=load_model, daemon=True).start()


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "time": time.time()
    }


@app.post("/complete")
async def complete(req: CompletionRequest):
    if model is None:
        return {"error": "Model not yet loaded"}
    inputs = tokenizer(req.prompt, return_tensors="pt")

    # offload blocking generate() to thread-pool
    with torch.inference_mode():
        outputs = await run_in_threadpool(lambda: model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        ))

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"completion": text[len(req.prompt):]}
