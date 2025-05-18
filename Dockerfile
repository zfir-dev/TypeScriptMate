FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

WORKDIR /code
RUN chmod a+rw /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app.py .

CMD uvicorn app:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --loop uvloop \
    --http httptools
