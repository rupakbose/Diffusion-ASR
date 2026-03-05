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
docker compose build
```

This will copy the mini dataset and its processed .pt files to the /app folder.

## Run Interactive Container

```bash
docker compose run --rm asr
```

You will get a bash prompt inside the container at /app.
All code, logs, and checkpoints are inside the container.

Once inside the interactive container shell, to run training,

```bash
root@xxxxxx:/app# python3 train.py
```

To run inference, change the audio path, steps, and the checkpoint in the inference.py.
Then run

```bash
root@xxxxxx:/app# python3 inference.py
```

## Training loss and metric curves

<div align="center">

<table>
<tr>
  <td align="center">
    <img src="./images/Loss_train.jpg" alt="Masked cross entropy loss" width="250"/><br/>
    <strong>Training Loss</strong>
  </td>
  <td align="center">
    <img src="./images/metric_CER.jpg" alt="CER" width="250"/><br/>
    <strong>Character Error Rate (CER)</strong>
  </td>
  <td align="center">
    <img src="./images/metric_WER.jpg" alt="WER" width="250"/><br/>
    <strong>Word Error Rate (WER)</strong>
  </td>
</tr>
</table>

</div>

<div align="center">
  <picture>
    <!-- Primary format: GIF -->
    <source srcset="./videos/inference.gif" type="image/gif">
    <!-- Fallback in case GIF is not supported -->
    <img src="https://github.com/rupakbose/Diffusion-ASR/blob/main/video/inference.gif" alt="ASR inference demo" width="500"/>
  </picture>
  <br/>
  <strong>Sample ASR Inference (Looping GIF)</strong>
</div>
