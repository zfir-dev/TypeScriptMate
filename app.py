import os
import time
import threading

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from starlette.concurrency import run_in_threadpool

torch.set_num_threads(8)
torch.set_num_interop_threads(2)

print("Starting app…")

app = FastAPI()

MODEL_PATH: str = None
tokenizer: AutoTokenizer = None
model: torch.nn.Module = None

HF_TOKEN = os.getenv("HF_TOKEN")


def load_model():
    global MODEL_PATH, tokenizer, model

    if HF_TOKEN:
        MODEL_PATH = snapshot_download(repo_id="zfir/TypeScriptMate", token=HF_TOKEN)
        print(f"Model files: {os.listdir(MODEL_PATH)}")
    else:
        MODEL_PATH = "model"
        print("No HF_TOKEN; using local ./model directory")

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model…")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).eval()

    try:
        print("Attempting dynamic int8 quantization…")
        model_q = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("✔ quantization succeeded")
    except Exception as e:
        print(f"⚠ quantization failed ({e}); using float32 model")
        model_q = base_model

    model = model_q
    model.eval()

    print("Warming up (1 token)…")
    dummy = tokenizer("Hi", return_tensors="pt")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=1)
    print("Warming up (40 tokens)…")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=40)
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

    start_time = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt")

    with torch.inference_mode():
        outputs = await run_in_threadpool(
            lambda: model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start_time
    print(f"Completion time: {elapsed:.3f} seconds")

    return {"completion": text[len(req.prompt):]}
