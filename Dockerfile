FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

WORKDIR /app

RUN pip install --no-cache-dir vllm

EXPOSE 7860

CMD ["bash", "-lc", "\
  vllm serve \
    --model zfir/TypeScriptMate \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --tensor-parallel-size 1 \
    --max-batch-size 8 \
    --max-batch-delay 10 \
"]
