FROM python:3.9

# Install Git + Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install

# Hugging Face auth for private model clone
# Expect HF_TOKEN in the build context
ARG HF_TOKEN
RUN git config --global credential.helper store && \
    echo "machine huggingface.co\nlogin zfir\npassword ${HF_TOKEN}" > ~/.netrc

WORKDIR /code

# Install Python deps
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Clone model repo
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/zfir/TypeScriptMate model && \
    cd model && git lfs pull

# Add your FastAPI app
COPY ./app.py /code/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
