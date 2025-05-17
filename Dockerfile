# Dockerfile
FROM ghcr.io/huggingface/text-generation-inference:3.3.0

# 1) Expose the port Spaces will map (defaults to $PORT=8000)
EXPOSE 8000

# 2) Use a writable cache (avoids any /root or /.cache permission errors)
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HOME=/tmp

# 3) Don’t override ENTRYPOINT (it’s already text-generation-server).
#    Just pass your flags via CMD in exec form:
CMD [ \
   "text-generation-server", \
   "--model-id",      "zfir/TypeScriptMate", \
   "--revision",      "main", \
   "--device",        "cpu", \
   "--disable-custom-kernels", \
   "--port",          "${PORT:-8000}" \
]
