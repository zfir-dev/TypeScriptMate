FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
