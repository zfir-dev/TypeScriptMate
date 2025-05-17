import threading, os, time, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Globals ────────────────────────────────────────────────────────────────
app = FastAPI()
MODEL_NAME = "zfir/TypeScriptMate"
HF_TOKEN   = os.getenv("HF_TOKEN")

tokenizer = None
model     = None

def load_model():
    global tokenizer, model
    logger.info("Downloading model snapshot…")
    local_dir = snapshot_download(repo_id=MODEL_NAME, token=HF_TOKEN)
    logger.info("Snapshot complete, files: %s", os.listdir(local_dir))

    logger.info("Loading tokenizer and model into memory…")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(local_dir).eval()
    logger.info("Model loaded.")

# kick off immediately, but non-blocking
threading.Thread(target=load_model, daemon=True).start()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

@app.post("/complete")
def complete(req: CompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model not yet loaded, please retry in a few seconds.")
    start = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, max_new_tokens=req.max_tokens, pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Request completed in %.2fs", time.time() - start)
    return {"completion": text[len(req.prompt):]}

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status":        "healthy",
        "model_loaded":  model is not None,
        "timestamp":     time.time(),
    }
