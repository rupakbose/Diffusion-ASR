# Diffusion ASR Training (Docker Compose)

This project contains a self-contained Docker environment for training an ASR (Automatic Speech Recognition) diffusion model. All dependencies, code, and logs live **inside the container**, so no host mounting is required.

You can run training interactively and save metrics/loss plots as SVG images.

---

## Prerequisites

- Linux with NVIDIA GPU + Docker + NVIDIA Container Toolkit
- `docker` & `docker-compose` installed
- CUDA 12.4 compatible GPU drivers

---

## Build Docker Image

```bash
# Build the Docker image using Docker Compose
docker compose build
```

---

## Run Interactive Container

```bash
# Start an interactive bash shell inside the container
docker compose run --rm asr


You will get a bash prompt inside the container at /app.
All code, logs, and checkpoints are inside the container.
GPU is available automatically.
```
