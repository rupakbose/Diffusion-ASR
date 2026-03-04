FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    libsndfile1 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# =========================================================

FROM base AS deps

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install -r requirements.txt

# =========================================================

FROM base AS runtime


COPY --from=deps /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin

WORKDIR /app

COPY . .

RUN mkdir -p /app/checkpoints /app/runs /root/.cache/huggingface

ENTRYPOINT ["/bin/bash"]