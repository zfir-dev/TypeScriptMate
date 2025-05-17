# 1) Base TGI image (pick a specific version or :latest)
FROM ghcr.io/huggingface/text-generation-inference:3.3.0

# 2) Expose the port that HF Spaces will map (defaults to 7860)
EXPOSE 8000

# 3) Point all Hugging Face caches at a writable directory
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

# 4) ENTRYPOINT is already set to text-generation-launcher in the base image.
#    We just pass our flags via CMD (shell form so ${PORT:-7860} expands).
CMD text-generation-launcher \
      --model-id zfir/TypeScriptMate \
      --revision main \
      --disable-custom-kernels \
      --port ${PORT:-8000}
