FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code/app.py

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --log-level ${LOG_LEVEL:-info}
