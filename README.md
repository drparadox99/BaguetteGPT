# Nietzsche-GPT2

An implementation and from-scratch training of a GPT-2-like model (124M parameters) on the complete writings of **Friedrich Nietzsche**, using the `krasaee/nietzsche` dataset from Hugging Face.

## 🧠 Overview

This project reproduces a ChatGPT2-style architecture (small GPT-2) and trains it entirely from scratch on Nietzsche’s corpus. The result is a model that generates text with a distinct philosophical and poetic tone, imitating Nietzsche's style.

## 📜 Highlights

- GPT-2 124M parameter architecture
- Trained from scratch (no pretrained weights)
- Dataset: [`krasaee/nietzsche`](https://huggingface.co/datasets/krasaee/nietzsche)
- Implemented using PyTorch & Hugging Face Transformers
- Capable of generating original philosophical prose in the style of Nietzsche

## 📂 Repository Structure



## 🗃 Dataset

- **Source**: [krasaee/nietzsche on Hugging Face](https://huggingface.co/datasets/krasaee/nietzsche)
- Preprocessing includes tokenization using `GPT2TokenizerFast`, lowercase conversion, and chunking into sequences of 1024 tokens.

## 🏗 Model

- **Architecture**: GPT-2 small (124M)
- **Specs**:
  - Layers: 12
  - Hidden size: 768
  - Heads: 12
  - Sequence length: 1024

## 🚀 Training

To train the model:

```bash
python train/train.py \
  --dataset krasaee/nietzsche \
  --model_config configs/gpt2_124M.json \
  --epochs 10 \
  --batch_size 4 \
  --lr 5e-5 \
  --output_dir checkpoints/


