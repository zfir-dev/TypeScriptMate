FROM python:3.9

# Install Git + Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install

# Accept Hugging Face token at build time
ARG HF_TOKEN

# Configure Git credentials for Hugging Face
RUN echo -e "machine huggingface.co\nlogin zfir\npassword ${HF_TOKEN}" > /root/.netrc

WORKDIR /code

# Install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Clone the model repo (with authentication)
RUN git clone https://huggingface.co/zfir/TypeScriptMate model && \
    cd model && git lfs pull

# Add FastAPI app
COPY ./app.py /code/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]