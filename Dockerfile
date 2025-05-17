FROM ghcr.io/huggingface/text-generation-inference:latest-cpu

# 1) Writable caches and HOME
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HOME=/tmp

WORKDIR /app

# 2) Expose the HF Spaces port
EXPOSE 8000

# 3) Supply only the TGI flags; the ENTRYPOINT is already `text-generation-launcher`
CMD ["--model-id", "zfir/TypeScriptMate", \
     "--revision", "main", \
     "--device", "cpu", \
     "--port", "${PORT:-8000}"]
