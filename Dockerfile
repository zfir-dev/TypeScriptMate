FROM ghcr.io/huggingface/text-generation-inference:latest

# Cache in a writable dir
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

# Use JSON array form for ENTRYPOINT, with backslashes to continue
ENTRYPOINT ["text-generation-server", \
    "--model-id", "zfir/TypeScriptMate", \
    "--revision", "main", \
    "--device", "cpu", \
    "--port", "${PORT:-8000}" \
]
