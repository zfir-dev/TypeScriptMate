import os
import time
import threading

os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
HF_TOKEN = os.getenv("HF_TOKEN")

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

torch.set_num_threads(8)
torch.set_num_interop_threads(2)

print("Starting app…")

app = FastAPI()

MODEL_PATH: str = None
tokenizer: AutoTokenizer = None
model: torch.nn.Module = None

def load_model():
    global MODEL_PATH, tokenizer, model

    if HF_TOKEN:
        MODEL_PATH = snapshot_download(
            repo_id="zfir/TypeScriptMate",
            token=HF_TOKEN
        )
        print(f"Model files: {os.listdir(MODEL_PATH)}")
    else:
        MODEL_PATH = "model"
        print("No HF_TOKEN; using local ./model directory")

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model…")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).eval()

    print("Quantizing model to int8…")
    model_q = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    try:
        print("Compiling model…")
        model = torch.compile(model_q)
        print("✔ compile succeeded")
    except Exception as e:
        print(f"⚠ compile failed ({e}); using quantized model")
        model = model_q

    model.eval()

    print("Warming up…")
    dummy = tokenizer("Hi", return_tensors="pt")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=1)
    print("Warm-up complete.")


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
        return {"error": "Model still loading…"}

    inputs = tokenizer(req.prompt, return_tensors="pt")
    with torch.inference_mode():
