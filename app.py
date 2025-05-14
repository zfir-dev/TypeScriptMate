# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
import gradio as gr
import uvicorn

# ─── Environment Config ────────────────────────────────────────────────────────
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

# ─── Gradio Interface ──────────────────────────────────────────────────────────
def generate_completion(prompt, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

gr_interface = gr.Interface(
    fn=generate_completion,
    inputs=[
        gr.Textbox(lines=4, label="Prompt"),
        gr.Slider(minimum=1, maximum=200, value=40, label="Max Tokens")
    ],
    outputs="text",
    title="TypeScriptMate API",
    description="Enter a prompt to get generated completions. Powered by a transformer model."
)

# ─── Conditional Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    if HF_TOKEN:
        gr_interface.launch(server_name="0.0.0.0", server_port=7860)
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
