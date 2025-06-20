# BaguetteGPT

BaguetteGPT is a transformer-based language model trained entirely from scratch on large-scale French-language datasets from Hugging Face, inspired by the GPT-2 architecture (124M parameters). This project demonstrates how to build a performant generative model for French using modern deep learning techniques, advanced optimization strategies, and distributed training at scale.

## 📜 Highlights

- GPT-2 124M parameter architecture  
- Trained from scratch (no pretrained weights)  
- **Datasets**: Hugging Face datasets, found in /data/loading_dataset.py 
- **Tokenizer**: OpenAI's open-source Tiktoken (Byte-pair encoding Tokenizer)
- **Training Framework**: PyTorch with Distributed Data Parallel (DDP) and SLURM
- **Checkpointing**: Intermediate and final checkpoints saved during training
- Implemented using PyTorch, Distributed Data Parallel (DDP) and Hugging Face Transformers  
- Capable of generating original philosophical prose in the style of Nietzsche  


## ⚙️ Training Features & Optimizations

This project incorporates numerous training techniques:

- 🖥 **Distributed Training**: PyTorch DistributedDataParallel (DDP)
- 🧠 **Mixed Precision Training**: Enabled via `torch.cuda.amp`
- 📉 **Gradient Accumulation**: To simulate large batch sizes
- 🕐 **Early Stopping**: With validation loss monitoring
- 🚀 **Flash Attention**: Optimized attention mechanism
- 🎯 **Fused Ops**: For faster computation
- 🔁 **Sampling Without Replacement**: For unbiased data shuffling
- 💡 **Learning Rate Scheduler**: Cosine decay with warmup
- ⬇️ **Weight Decay**
- 📉 **Decaying LR + AdamW Betas + Gradient Clipping**
- 💾 **Torch Compile**: Leveraged for graph-level optimization


## 🛠 Training Environment

- **Cluster**: SLURM-managed HPC cluster
- **Accelerators**: NVIDIA A100 (or similar)
- **Training Time**: ~X hours over Y GPUs (update based on logs)
- **Frameworks**: PyTorch, HuggingFace Datasets & Tokenizers
  

## 📜 ChatGPT-Inspired Hyperparameters

| Hyperparameter        | Value                       |
|-----------------------|-----------------------------|
| Model Dim             | 768                         |
| Layers                | 12                          |
| Heads                 | 12                          |
| Context Length        | 1024                        |
| Batch Size (Global)   | Variable via Accumulation   |
| LR Scheduler          | Cosine w/ Warmup            |
| Base Learning Rate    | 3e-4                        |
| Weight Decay          | 0.1                         |
| Adam Betas            | (0.9, 0.95)                 |
| Gradient Clipping     | 1.0                         |
| Precision             | Mixed (float16)             |
| Epochs                | Set via early stopping      |

## 📦 Requirements

Install dependencies:

```bash
conda env create -f environment.yml
```

## 🚀 Usage

Download and tokenize datasets: 

python data/loading_dataset.py

To train the model:

  - bash run.sh

Supports training on single or multiple GPUs via `torch.distributed`.

Key libraries:
- `transformers`
- `datasets`
- `torch`
- `tiktoken`

## 📈 Future Work

- Add top-k sampling  
- Fine-tune on more/other philosophical texts  
- Integrate LoRA or MoE variants  
- Deploy with a simple web UI (e.g. Gradio)  

## 📚 Citation & References

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya and others},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}

## 🤝 Acknowledgments

- Inspired by Andrej Karpathy's build nanoGPT project [Contributor's GitHub Page](https://github.com/karpathy/build-nanogpt.git) 
- Datasets by [https://huggingface.co](https://huggingface.co)

## 📖 Licence 

This project is licensed under the MIT License - see the LICENSE file for details.



