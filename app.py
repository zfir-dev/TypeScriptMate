from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "model"
MAX_INPUT_TOKENS = 512

# ─── Device Setup ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load Model & Tokenizer ───────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Optional: quantize the model for faster CPU inference
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
if device.type == "cpu":
    print("Quantizing model for CPU...")
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

model.to(device)
model.eval()
print("Model loaded.")

# ─── FastAPI Setup ────────────────────────────────────────────────────────────
app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

# ─── Warm-up (first-time generation is slow due to graph compile) ─────────────
@app.on_event("startup")
def warm_up_model():
    print("Warming up model...")
    dummy_prompt = "function hello() {"
    inputs = tokenizer(dummy_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id
        )
    print("Warm-up complete.")

# ─── Inference Route ──────────────────────────────────────────────────────────
@app.post("/complete")
def complete(req: CompletionRequest):
    start = time.time()

    # Tokenize input with truncation
    inputs = tokenizer(
        req.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output[len(req.prompt):]
    duration = time.time() - start
    print(f"Completed request in {duration:.2f}s")

    return {"completion": generated.strip()}
