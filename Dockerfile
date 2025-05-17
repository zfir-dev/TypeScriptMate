FROM python:3.9

ENV HF_HOME=/tmp/hf_cache

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code/app.py

RUN rm -rf /tmp/hf_cache

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
