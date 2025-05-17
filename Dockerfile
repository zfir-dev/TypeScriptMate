FROM python:3.9-slim

ENV LOGNAME=root \
    USER=root

ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HOME=/tmp/hf_cache \
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
    --enforce-eager \
    --disable-async-output-proc \
    --max-num-seqs 8 \
    --scheduler-delay-factor 0.01 \
"
