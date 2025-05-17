FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    LOGNAME=root \
    USER=root \
    TORCHINDUCTOR_CACHE_DIR=/tmp/hf_cache

WORKDIR /app
RUN pip install --no-cache-dir "vllm[cpu]"

EXPOSE 8000

CMD bash -lc "\
  vllm serve \
    zfir/TypeScriptMate \
    --device cpu \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --distributed-executor-backend none \
    --enforce-eager \
    --disable-async-output-proc \
    --max-num-seqs 8 \
    --scheduler-delay-factor 0.01 \
"
