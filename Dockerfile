# =============================================================================
# autotrain — CUDA image
#
# Two stages. The `base` stage carries only the data-pipeline dependencies and
# is enough for phases 1-7 (crawl -> split); it is small and CPU-only. The
# `cuda` stage adds torch, PEFT, TRL and bitsandbytes for phase 8 onward.
#
# Build the full image (default):
#   docker build -t autotrain:cuda .
#
# Build the lightweight data-only image:
#   docker build --target base -t autotrain:data .
#
# Run against a GPU host:
#   docker run --gpus all -v "$PWD/runs:/work/runs" autotrain:cuda https://example.com
#
# Run data phases only, no GPU needed:
#   docker run -v "$PWD/runs:/work/runs" autotrain:data https://example.com --skip-training
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — data pipeline (phases 1-7)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# lxml needs libxml2/libxslt headers at build time; curl is used by the
# healthcheck and is genuinely useful when debugging a crawl from inside.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy dependency metadata first so the layer cache survives source edits.
COPY pyproject.toml README.md ./
COPY src/autotrain/__init__.py src/autotrain/__init__.py
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY configs/ configs/
COPY main.py ./

# Non-root by default. The runs/ mount must be writable by this uid.
RUN useradd --create-home --uid 1000 autotrain \
    && mkdir -p /work/runs \
    && chown -R autotrain:autotrain /work
USER autotrain

ENV HF_HOME=/work/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]


# -----------------------------------------------------------------------------
# Stage 2 — training (phases 8-12)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /work

COPY pyproject.toml README.md ./
COPY src/autotrain/__init__.py src/autotrain/__init__.py

# torch first and on its own layer: it is by far the largest dependency, and
# keeping it separate means a change to the training extras does not force a
# multi-gigabyte re-download.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -e ".[train,cuda,eval]"

COPY src/ src/
COPY configs/ configs/
COPY main.py ./
COPY tests/ tests/

RUN useradd --create-home --uid 1000 autotrain \
    && mkdir -p /work/runs /work/.cache \
    && chown -R autotrain:autotrain /work
USER autotrain

ENV HF_HOME=/work/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false

# Fails fast on a broken image rather than three hours into a crawl.
HEALTHCHECK --interval=60s --timeout=15s --start-period=10s --retries=2 \
    CMD python -c "import autotrain, torch; print(torch.cuda.is_available())" || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
