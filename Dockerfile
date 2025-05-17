FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf_cache

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code/app.py

EXPOSE 7860
CMD ["python", "app.py"]
