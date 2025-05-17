FROM ghcr.io/huggingface/text-generation-inference:latest

EXPOSE 8000

ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

ENTRYPOINT ["text-generation-server", 
    "--model-id", "zfir/TypeScriptMate",
    "--revision", "main",
    "--device", "cpu",
    "--port", "${PORT:-7860}"
]
