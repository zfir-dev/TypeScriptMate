from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForCausalLM.from_pretrained("model")
model.eval()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/complete")
def complete(request: CompletionRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"completion": completion[len(request.prompt):]}
