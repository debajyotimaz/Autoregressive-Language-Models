# Autoregressive Language Models: RNN vs LSTM vs GRU

> **Tutorial for NLP Jan 2026 (DSE 318/401/607) | Tuesday, 24th March 2026**  
> Prepared by **Suraj** and **Debajyoti** 

---

## What This Tutorial Covers

This repository provides a **hands-on, end-to-end tutorial** on building **Autoregressive Language Models** using three recurrent architectures — **RNN**, **LSTM**, and **GRU** — trained on a real Wikipedia text corpus using PyTorch.

By the end of this tutorial, participants will be able to:

1. Implement all three architectures from scratch in PyTorch
2. Train autoregressive language models on Wikipedia data
3. Evaluate models using **Perplexity**
4. Generate text autoregressively from each trained model
5. Compare architectures **quantitatively** (metrics) and **qualitatively** (generated text)

---

## Quick Start — Run in Google Colab

Click the badge to open a notebook directly in Colab:

| Notebook | Description |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/debajyotimaz/Autoregressive-Language-Models/blob/main/training_by_teacher_forcing.ipynb) | **Teacher Forcing** — standard training loop |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/debajyotimaz/Autoregressive-Language-Models/blob/main/training_by_autoreg.ipynb) | **Autoregressive Training** — model feeds its own predictions |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/debajyotimaz/Autoregressive-Language-Models/blob/main/training_by_teacher_forcing.ipynb) | **Hybrid Training** — Use both |

> **Recommended:** Use a **GPU runtime** in Colab (`Runtime → Change runtime type → T4 GPU`) for faster training.

---

## Architecture Overview

| Architecture | Gates | Memory | Characteristic |
|---|---|---|---|
| **RNN** | — (tanh only) | Hidden state only | Suffers from vanishing gradients |
| **LSTM** | Forget, Input, Output | Cell state + hidden | Long-range memory |
| **GRU** | Update, Reset | Hidden state | fewer params than LSTM |

---

## Dataset

We use the **WikiText-2** dataset — a small, clean, representative subset of Wikipedia articles:
- ~2 million training tokens
- ~200K validation tokens  
- ~200K test tokens
- English Wikipedia articles filtered for high quality
- Available via `torchtext` (automatically downloaded in the notebook)

---

## 🛠 Requirements

```
torch>=2.0.0
torchtext>=0.15.0
datasets>=2.0.0
numpy
matplotlib
tqdm
sacrebleu
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📖 References

1. Jurafsky, D., & Martin, J. H. (2026). Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models (3rd edn).
