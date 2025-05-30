import os
import time
import threading
import csv

import torch
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from starlette.concurrency import run_in_threadpool
from supabase import create_client, Client
from peft import PeftModel, PeftConfig

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "zfir/TypeScriptMate")
HF_TOKEN = os.getenv("HF_TOKEN")
USE_LORA = bool(int(os.getenv("USE_LORA", "0")))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

BUCKET = "typescriptmate"
FEEDBACK_LOG = "feedbacks.csv"
COMPLETION_LOG = "completions.csv"

for log_name in (COMPLETION_LOG, FEEDBACK_LOG):
    try:
        res = supabase.storage.from_(BUCKET).download(log_name)
        data = res.content if hasattr(res, "content") else res
        with open(log_name, "wb") as f:
            f.write(data)
        print(f"Loaded existing {log_name} from Supabase")
    except Exception:
        print(f"No existing {log_name} in bucket; starting fresh")

torch.set_num_threads(8)
torch.set_num_interop_threads(2)

print("Starting app…")

app = FastAPI()

MODEL_PATH: str = None
tokenizer: GPT2TokenizerFast = None
model: torch.nn.Module = None

def write_feedback_log(event: dict):
    is_new = not os.path.isfile(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt", "model", "completion", "action", "timestamp"]
        )
        if is_new:
            writer.writeheader()
        writer.writerow(event)
    
    if supabase:
        with open(FEEDBACK_LOG, "rb") as file_obj:
            try:
                supabase.storage.from_(BUCKET).upload(
                    FEEDBACK_LOG,
                    file_obj,
                    file_options={"upsert": "true"},
                )
            except Exception as e:
                print("Failed to upload feedback log:", e)

def write_completion_log(event: dict):
    file_exists = os.path.isfile(COMPLETION_LOG)
    with open(COMPLETION_LOG, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["prompt", "completion", "latency_s", "timestamp"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(event)
        
    if supabase:
        with open(COMPLETION_LOG, "rb") as file_obj:
            try:
                supabase.storage.from_(BUCKET).upload(
                    COMPLETION_LOG,
                    file_obj,
                    file_options={"upsert": "true"},
                )
            except Exception as e:
                print("Failed to upload completion log:", e)


def load_model():
    global tokenizer, model

    if HF_TOKEN and MODEL_REPO_ID:
        MODEL_PATH = snapshot_download(repo_id=MODEL_REPO_ID, token=HF_TOKEN)
        print(f"Model files: {os.listdir(MODEL_PATH)}")
    else:
        MODEL_PATH = "model"
        print("No HF_TOKEN; using local ./model directory")

    if USE_LORA:
        print("Loading LoRA model…")

        print("Loading adapter config…")
        adapter_config = PeftConfig.from_pretrained(MODEL_PATH)

        base_model_name_or_path = adapter_config.base_model_name_or_path

        print("Loading tokenizer…")
        tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        print("Loading base model…")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, MODEL_PATH, local_files_only=True)
    else:
        print("Loading tokenizer…")
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        print("Loading base model…")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).eval()
        model = base_model

    model.eval()

    print("Warming up (1 token)…")
    dummy = tokenizer("console.log", return_tensors="pt")
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

class Feedback(BaseModel):
    model: str
    prompt: str
    completion: str
    action: str
    timestamp: float

@app.get("/")
@app.get("/health")
def index_and_health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "time": time.time()
    }


@app.get("/logs", response_class=HTMLResponse)
def logs():
    def read_last_rows(path: str, max_rows: int = 20):
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        except FileNotFoundError:
            return [], []

        if not rows:
            return [], []

        header, *entries = rows
        last = entries[-max_rows:] if len(entries) > max_rows else entries
        return header, last

    comp_header, comp_rows = read_last_rows(COMPLETION_LOG)
    fb_header, fb_rows     = read_last_rows(FEEDBACK_LOG)

    html = ["<html><body style='font-family: sans-serif'>"]

    if comp_header:
        html.append("<h2>Last 20 Completions</h2>")
        html.append("<table border='1' style='border-collapse:collapse;margin-bottom:2em'>")
        html.append("<thead><tr>" + "".join(f"<th style='padding:4px'>{col}</th>" for col in comp_header) + "</tr></thead>")
        html.append("<tbody>")
        for row in comp_rows:
            html.append("<tr>" + "".join(f"<td style='padding:4px'>{cell}</td>" for cell in row) + "</tr>")
        html.append("</tbody></table>")
    else:
        html.append("<p><em>No completion logs found</em></p>")

    if fb_header:
        html.append("<h2>Last 20 Feedbacks</h2>")
        html.append("<table border='1' style='border-collapse:collapse'>")
        html.append("<thead><tr>" + "".join(f"<th style='padding:4px'>{col}</th>" for col in fb_header) + "</tr></thead>")
        html.append("<tbody>")
        for row in fb_rows:
            html.append("<tr>" + "".join(f"<td style='padding:4px'>{cell}</td>" for cell in row) + "</tr>")
        html.append("</tbody></table>")
    else:
        html.append("<p><em>No feedback logs found</em></p>")

    html.append("</body></html>")
    return HTMLResponse("".join(html))

@app.post("/complete")
async def complete(
    req: CompletionRequest,
    background_tasks: BackgroundTasks
):
    if model is None:
        return {"error": "Model still loading…"}

    start = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = await run_in_threadpool(
            lambda: model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_len:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    latency = time.time() - start
    event = {
        "prompt": req.prompt,
        "model": MODEL_REPO_ID if MODEL_REPO_ID else "local",
        "completion": completion,
        "latency_s": latency,
        "timestamp": time.time(),
    }

    background_tasks.add_task(write_completion_log, event)

    return {"completion": completion}

@app.post("/feedbacks")
def feedback_endpoint(ev: Feedback, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_feedback_log, {
        "prompt": ev.prompt,
        "model": MODEL_REPO_ID if MODEL_REPO_ID else "local",
        "completion": ev.completion,
        "action": ev.action,
        "timestamp": ev.timestamp,
    })
    return {"status": "ok"}
