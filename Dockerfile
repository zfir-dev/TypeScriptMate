# Dockerfile
FROM ghcr.io/huggingface/text-generation-inference:latest-cpu

# 1) Expose the Spaces port
EXPOSE 8000

# 2) Point HF caches at a writable dir
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

# 3) ENTRYPOINT is already set to text-generation-server in the base image.
#    We just pass our flags via CMD:
CMD text-generation-server \
    --model-id zfir/TypeScriptMate \
    --revision main \
    --device cpu \
    --port ${PORT:-8000}
