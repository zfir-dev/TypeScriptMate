import os
import time
import threading
import csv
import uuid
import json
from typing import Union, List, Optional

import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from starlette.concurrency import run_in_threadpool
from supabase import create_client, Client
from peft import PeftConfig, PeftModel

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")
USE_LORA = bool(int(os.getenv("USE_LORA", "0")))
USE_QUANTIZATION = bool(int(os.getenv("USE_QUANTIZATION", "1")))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client connected")
else:
    supabase = None
    print("Supabase client not connected")

BUCKET = "typescriptmate"
FEEDBACK_LOG = "feedbacks.csv"
MODIFIED_FEEDBACK_LOG = "feedbacks.modified.csv"
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
    file_exists = os.path.isfile(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                # Base fields
                "timestamp", "userId", "userAgent", "selectedProfileId", 
                "eventName", "schema",
                # Autocomplete fields
                "disable", "maxPromptTokens", "debounceDelay", 
                "maxSuffixPercentage", "prefixPercentage", "transform",
                "template", "multilineCompletions", "slidingWindowPrefixPercentage",
                "slidingWindowSize", "useCache", "onlyMyCode", "useRecentlyEdited",
                "useImports", "accepted", "time", "prefix", "suffix",
                "prompt", "completion", "modelProvider", "modelName",
                "cacheHit", "filepath", "gitRepo", "completionId", "uniqueId",
                # Metadata fields
                "event_name", "schema_version", "level", "profile_id"
            ]
        )
        if not file_exists:
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
            fieldnames=["prompt", "model", "completion", "latency_s", "timestamp"]
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
    global tokenizer, model, MODEL_PATH

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
            torch_dtype=torch.float32        
        )
        model = PeftModel.from_pretrained(
            base_model, 
            MODEL_PATH, 
            torch_dtype=torch.float32, 
            local_files_only=True
        )
        model = model.merge_and_unload()

    else:
        print("Loading vanilla model…")
        
        print("Loading tokenizer…")
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        print("Loading base model…")
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    print("Supported quantization engines:", torch.backends.quantized.supported_engines)
    torch.backends.quantized.engine = 'qnnpack'

    if USE_QUANTIZATION:
        print("Quantizing model…")
        model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )

    model.eval()

    print("Warming up (1 token)…")
    dummy = tokenizer("console.log", return_tensors="pt")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=1)
    print("Warming up (40 tokens)…")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=40)
    print("Warming up (100 tokens)…")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=100)
    print("Warming up (200 tokens)…")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=200)
    print("Warming up (400 tokens)…")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=400)
    print("Warming up (800 tokens)…")
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=800)
    print("Warm-up complete.")


threading.Thread(target=load_model, daemon=True).start()


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 40

class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(..., description="Either a string or a list of strings")
    max_tokens: int = 40
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None

class ContinueAutocompleteData(BaseModel):
    timestamp: float
    userId: Optional[str] = None
    userAgent: Optional[str] = None
    selectedProfileId: Optional[str] = None
    eventName: Optional[str] = None
    schema: Optional[str] = None
    disable: Optional[bool] = None
    maxPromptTokens: Optional[int] = None
    debounceDelay: Optional[int] = None
    maxSuffixPercentage: Optional[float] = None
    prefixPercentage: Optional[float] = None
    transform: Optional[Union[bool, str]] = None
    template: Optional[str] = None
    multilineCompletions: Optional[Union[bool, str]] = None
    slidingWindowPrefixPercentage: Optional[float] = None
    slidingWindowSize: Optional[int] = None
    useCache: Optional[bool] = None
    onlyMyCode: Optional[bool] = None
    useRecentlyEdited: Optional[bool] = None
    useImports: Optional[bool] = None
    accepted: Optional[bool] = None
    time: Optional[float] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    modelProvider: Optional[str] = None
    modelName: Optional[str] = None
    cacheHit: Optional[bool] = None
    filepath: Optional[str] = None
    gitRepo: Optional[str] = None
    completionId: Optional[str] = None
    uniqueId: Optional[str] = None

class ContinueFeedback(BaseModel):
    name: str
    data: ContinueAutocompleteData
    schema: str
    level: Optional[str] = None
    profileId: Optional[str] = None

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
    fb_header, fb_rows = read_last_rows(MODIFIED_FEEDBACK_LOG)

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

def get_max_sequence_length():
    if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
        return model.config.max_position_embeddings
    elif hasattr(model, 'config') and hasattr(model.config, 'n_positions'):
        return model.config.n_positions
    else:
        return 1024

def truncate_prompt_if_needed(prompt: str, max_tokens: int = 40) -> str:
    max_seq_len = get_max_sequence_length()
    max_prompt_len = max_seq_len - max_tokens - 10
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if inputs["input_ids"].shape[-1] > max_prompt_len:
        input_ids = inputs["input_ids"][0][-max_prompt_len:]
        truncated_prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Warning: Prompt truncated from {inputs['input_ids'].shape[-1]} to {len(input_ids)} tokens")
        return truncated_prompt
    return prompt

@app.post("/v1/completions")
async def complete(
    req: OpenAICompletionRequest,
    background_tasks: BackgroundTasks
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading…")

    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    
    truncated_prompts = [truncate_prompt_if_needed(prompt, req.max_tokens) for prompt in prompts]

    start_all = time.time()
    
    if req.stream:
        async def generate_stream():
            for idx, single_prompt in enumerate(truncated_prompts):
                inputs = tokenizer(single_prompt, return_tensors="pt")
                prompt_len = inputs["input_ids"].shape[-1]
                
                for choice_idx in range(req.n):
                    with torch.inference_mode():
                        outputs = await run_in_threadpool(
                            lambda: model.generate(
                                **inputs,
                                max_new_tokens=req.max_tokens,
                                pad_token_id=tokenizer.eos_token_id,
                                temperature=req.temperature,
                                top_p=req.top_p,
                                do_sample=(req.temperature != 0.0 or req.top_p < 1.0),
                                return_dict_in_generate=True,
                                output_scores=True,
                            )
                        )
                    
                    generated_ids = outputs.sequences[0][prompt_len:]
                    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    elapsed = time.time() - start_all

                    event = {
                        "prompt": single_prompt,
                        "model": MODEL_REPO_ID if MODEL_REPO_ID else "local",
                        "completion": completion_text,
                        "latency_s": elapsed,
                        "timestamp": time.time(),
                    }
                    background_tasks.add_task(write_completion_log, event)
                    
                    response = {
                        "id": str(uuid.uuid4()),
                        "object": "text_completion",
                        "created": int(time.time()),
                        "choices": [{
                            "text": completion_text,
                            "index": float(idx * req.n + choice_idx),
                            "logprobs": None,
                            "finish_reason": "length"
                        }],
                        "model": req.model
                    }
                    yield f"data: {json.dumps(response)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    all_choices = []
    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    for idx, single_prompt in enumerate(truncated_prompts):
        inputs = tokenizer(single_prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[-1]
        usage_prompt_tokens += prompt_len

        for choice_idx in range(req.n):
            with torch.inference_mode():
                outputs = await run_in_threadpool(
                    lambda: model.generate(
                        **inputs,
                        max_new_tokens=req.max_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        do_sample=(req.temperature != 0.0 or req.top_p < 1.0),
                    )
                )

            generated_ids = outputs[0][prompt_len:]
            num_generated = generated_ids.shape[0]
            usage_completion_tokens += num_generated

            completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_choices.append({
                "text": completion_text,
                "index": float(idx * req.n + choice_idx),
                "logprobs": None,
                "finish_reason": "length"
            })

    total_tokens = usage_prompt_tokens + usage_completion_tokens
    elapsed = time.time() - start_all

    event = {
        "prompt": truncated_prompts[0],
        "model": MODEL_REPO_ID if MODEL_REPO_ID else "local",
        "completion": all_choices[0]["text"],
        "latency_s": elapsed,
        "timestamp": time.time(),
    }
    background_tasks.add_task(write_completion_log, event)

    response = {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": all_choices,
        "usage": {
            "prompt_tokens": usage_prompt_tokens,
            "completion_tokens": usage_completion_tokens,
            "total_tokens": total_tokens
        }
    }
    return response

@app.post("/complete")
async def legacy_complete(
    req: CompletionRequest,
    background_tasks: BackgroundTasks
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading…")

    truncated_prompt = truncate_prompt_if_needed(req.prompt, req.max_tokens)

    start = time.time()
    inputs = tokenizer(truncated_prompt, return_tensors="pt")
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


@app.post("/feedback")
async def feedback(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        
        try:
            ev = ContinueFeedback(**body)
        except Exception as e:
            print("Validation error:", str(e))
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Validation error",
                    "message": str(e),
                    "received_data": body
                }
            )
        
        event = ev.data.dict(exclude_none=True)
        
        event.update({
            "event_name": ev.name,
            "schema_version": ev.schema,
            "level": ev.level,
            "profile_id": ev.profileId,
            "modelName": MODEL_REPO_ID if MODEL_REPO_ID else "local",
        })
        
        background_tasks.add_task(write_feedback_log, event)
        return {"status": "ok"}
        
    except json.JSONDecodeError as e:
        print("JSON decode error:", str(e))
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid JSON",
                "message": str(e)
            }
        )
    except Exception as e:
        print("Unexpected error:", str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e)
            }
        )
