# Automatic speech recognition using Diffusion language models

Traditional ASR models like Zipformer, Conformer, ESPNet, Whisper and wav2vec are sequence to sequence models.
They are autoregressive models that predict the next token in the sequence.
Such models have a bottleneck with inference speeds when transcribing long sentences.

[Large Language Diffusion Models](https://doi.org/10.48550/arXiv.2502.09992) (LLaDA) [18 oct 2025], is a diffusion model that generates language as a probabilistic inference.
It's a masked language model that uses iterative denoising process to generate tokens parallel in contrast to sequence to sequence models. It is pretrained for masked prediction by a transformer model.

Once pretrained, it follows a supervised finetuning (SFT) where the dataset is in the form of prompts and responses.
The responses are masked using a probability distribution and then the loss is computed only on the masked predictions.
During the inference process, we start with a completely masked response and iteratively denoise the predicted tokens over sampling steps.

In this project, we adapt LLaDa's supervised finetuning protocol for ASR, and generate transcriptions using denoising process.
We utilise audio features in place of prompts, and transcripts in place od responses and do masked modelling.
The architecture is as follows:

<div align="center">
    <img src="./images/archi.jpg" alt="archi" width="500" />
</div>

Additionally, we adapt the scaled loss in Algorithm 2 to optimise our network and showcase that transcriiption is possible using diffusion protocol.

<div align="center">
    <img src="./images/algo2.jpg" alt="Algo" width="500" />
</div>

## Pipeline and Contributions

### 1. Feature Extraction

The process begins by passing the raw audio through a **wav2vec** model to extract high-level acoustic representations. This results in an initial feature set $audio\_features$ with dimensions `[batch_size, sequence_length, 768]`.

### 2. Temporal Sampling and Padding

To maintain temporal coherence while managing computational load, we process the audio features by:

- Randomly sampling `window_size` indices where the index $i < \text{sequence\_length}$.
- Sorting these indices to preserve the chronological flow of the audio.
- Constructing a source tensor $src$ of shape `[batch_size, window_size, 768]`.

> **Design Choice:** This method effectively drops redundant features while ensuring the remaining data points are temporally aligned.

### 3. Text Tokenization

The target transcription is tokenized and padded to a fixed `transcription_length` to generate the `input_ids`. These IDs serve as the ground truth for our denoising process.

### 4. Bernoulli Masking and Model Input

Following a **Bernoulli distribution**, we apply a masking strategy to the `input_ids` to create `masked_ids`.

- The transformer model receives both the sampled audio features ($src$) and the $masked\_ids$ as inputs.
- **Custom Architecture:** We implemented a complete **Transformer model** from scratch, including:
  - Multi-head Attention mechanisms.
  - Sinusoidal Positional Embeddings to capture sequence order.

### 5. Optimization via Scaled Loss

The model is tasked with predicting the original tokens beneath the mask.

- Loss is computed between the predicted masked tokens and the ground truth.
- **Algorithm Implementation:** We integrated the **scaled loss logic** from **Algorithm 2** to optimize the network specifically for the diffusion-based fine-tuning protocol.

### 6. Training Stability

To ensure robust convergence and prevent gradient explosions during the diffusion process, we utilize **gradient clipping** as a final step in the optimization pipeline.

---

### Implementation Highlights

| Component             | Status          | Description                                               |
| :-------------------- | :-------------- | :-------------------------------------------------------- |
| **Transformer Model** | **Custom**      | Full implementation of Attention and Positional Encoding. |
| **Masking Strategy**  | **Custom**      | Bernoulli-based masking for diffusion denoising.          |
| **Scaled Loss**       | **Algorithm 2** | Adapted from LLaDA for Supervised Fine-Tuning (SFT).      |

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

## ASR inference using 3000 denoising steps

![](https://github.com/rupakbose/Diffusion-ASR/blob/main/video/inference.gif)

## ASR inference using 100 denoising steps

![](https://github.com/rupakbose/Diffusion-ASR/blob/main/video/inference100.gif)
