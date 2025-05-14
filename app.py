# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
import gradio as gr
import uvicorn

MODEL = os.getenv("MODEL_NAME", "model")
HF_TOKEN = os.getenv("HF_TOKEN")

# ─── Load model & tokenizer ────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL, token=HF_TOKEN)
model.eval()
print("Model loaded.")

# ─── FastAPI Setup ────────────────────────────────────────────────────────────
app = FastAPI(
    title="TypeScriptMate API",
    description="API for TypeScript code generation and completion",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

def generate_text(prompt: str, max_tokens: int = 40) -> str:
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Completed request in {time.time() - start:.2f}s")
    return result[len(prompt):]

# Create Gradio interface if HF_TOKEN is present
if HF_TOKEN:
    def gradio_generate(prompt: str, max_tokens: int = 40) -> str:
        return generate_text(prompt, max_tokens)

    gradio_interface = gr.Interface(
        fn=gradio_generate,
        inputs=[
            gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here..."),
            gr.Slider(minimum=1, maximum=200, value=40, label="Max Tokens")
        ],
        outputs=gr.Textbox(label="Completion"),
        title="TypeScriptMate API",
        description=f"Using model: {MODEL}"
    )
    app = gr.mount_gradio_app(app, gradio_interface, path="/")

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

    class Config:
        schema_extra = {
            "example": {
                "prompt": "function calculateSum(a: number, b: number): number {",
                "max_tokens": 40
            }
        }

class CompletionResponse(BaseModel):
    completion: str
    model: str = MODEL

@app.post("/api/complete", response_model=CompletionResponse)
async def complete(req: CompletionRequest):
    """
    Generate TypeScript code completion based on the input prompt.
    
    Args:
        req (CompletionRequest): The request containing prompt and generation parameters
        
    Returns:
        CompletionResponse: The generated completion and model information
    """
    try:
        completion = generate_text(req.prompt, req.max_tokens)
        return CompletionResponse(completion=completion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and get model information.
    
    Returns:
        dict: API status and model information
    """
    return {
        "status": "healthy",
        "model": MODEL,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
