# Use the official TGI image (has correct ENTRYPOINT & permissions baked in)
FROM ghcr.io/huggingface/text-generation-inference:latest

# Expose the port that Spaces will map (7860)
EXPOSE 8000

# Cache in a writable dir
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

# *Do not* override ENTRYPOINT!
# Just supply the args via CMD so the image's own entrypoint (text-generation-server) runs
CMD ["--model-id", "zfir/TypeScriptMate", \
     "--revision", "main", \
     "--device", "cpu", \
     "--port", "${PORT:-8000}"]
