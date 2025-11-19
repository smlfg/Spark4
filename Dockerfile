# Production-Ready Dockerfile for Unsloth Training
# Base: NVIDIA PyTorch 24.01 (Python 3.10, CUDA 12.3, PyTorch 2.2.0)
# All versions are pinned for reproducibility

FROM nvcr.io/nvidia/pytorch:24.01-py3 AS builder

# Metadata
LABEL maintainer="Spark4"
LABEL description="Production Unsloth training environment with pinned dependencies"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    MAX_JOBS=4

# Update system packages and install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0

# Install packaging tools with pinned versions
RUN pip install --no-cache-dir \
    packaging==24.0 \
    ninja==1.11.1.1

# Pre-compile Flash Attention 2 (this takes time, so we do it early for caching)
RUN pip install --no-cache-dir flash-attn==2.5.9.post1 --no-build-isolation

# Install core ML dependencies with pinned versions
# These versions are tested to work together with Unsloth
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    transformers==4.45.2 \
    tokenizers==0.20.3 \
    accelerate==0.33.0 \
    peft==0.13.2 \
    bitsandbytes==0.44.1 \
    datasets==2.20.0 \
    trl==0.11.4 \
    sentencepiece==0.2.0 \
    protobuf==5.27.2

# Install Unsloth with pinned version
# Using 2024.11 series for stability with the above dependencies
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@4e11ceb3b0eb486764c2f5f47c8eb48e7ce0c8e1"

# Install additional utilities
RUN pip install --no-cache-dir \
    wandb==0.18.7 \
    tensorboard==2.16.2 \
    scikit-learn==1.5.1 \
    scipy==1.14.1 \
    matplotlib==3.9.2 \
    pandas==2.2.2 \
    tqdm==4.66.5 \
    huggingface-hub==0.25.2

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import trl; print(f'TRL: {trl.__version__}')" && \
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" && \
    python -c "import unsloth; print('Unsloth installed successfully')" && \
    python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')"

# Production stage
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Copy environment variables from builder
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app /app/.cache/huggingface && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy verification script
COPY --chown=appuser:appuser verify_environment.py /app/verify_environment.py

# Switch to non-root user
USER appuser

# Verify installations as non-root user
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import unsloth; print('Unsloth ready')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["/bin/bash"]
