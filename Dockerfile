FROM python:3.9

ENV PYTHONUNBUFFERED=1  

ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app.py .

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info --workers 4
