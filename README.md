# 🧠 Autoregressive Language Models: RNN vs LSTM vs GRU

> **Tutorial for NLP Lab | Tuesday, 24th March 2026**  
> Prepared by **Suraj** and **Debajyoti** 

---

## 📌 What This Tutorial Covers

This repository provides a **hands-on, end-to-end tutorial** on building **Autoregressive Language Models** using three recurrent architectures — **Vanilla RNN**, **LSTM**, and **GRU** — trained on a real Wikipedia text corpus using PyTorch.

By the end of this tutorial, participants will be able to:

1. Understand the theory of RNNs, LSTMs, and GRUs
2. Implement all three architectures from scratch in PyTorch
3. Train autoregressive language models on Wikipedia data
4. Evaluate models using **Perplexity** and **BLEU score**
5. Generate text autoregressively from each trained model
6. Compare architectures **quantitatively** (metrics) and **qualitatively** (generated text)

---

## 🚀 Quick Start — Run in Google Colab

Click the badge to open a notebook directly in Colab:

| Notebook | Description |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/debajyotimaz/Autoregressive-Language-Models/blob/main/AutoregressiveLM_Tutorial.ipynb) | **Teacher Forcing** — standard training loop |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/debajyotimaz/Autoregressive-Language-Models/blob/main/AutoregressiveLM_Tutorial_ARTrain.ipynb) | **Autoregressive Training** — model feeds its own predictions |

> **Recommended:** Use a **GPU runtime** in Colab (`Runtime → Change runtime type → T4 GPU`) for faster training.

---

## 📚 Architecture Overview

| Architecture | Gates | Memory | Characteristic |
|---|---|---|---|
| **Vanilla RNN** | — (tanh only) | Hidden state only | Simple; suffers from vanishing gradients |
| **LSTM** | Forget, Input, Output | Cell state + hidden | Best long-range memory |
| **GRU** | Update, Reset | Hidden state | ~25% fewer params than LSTM |

---

## 📊 Dataset

We use the **WikiText-2** dataset — a small, clean, representative subset of Wikipedia articles:
- ~2 million training tokens
- ~200K validation tokens  
- ~200K test tokens
- English Wikipedia articles filtered for high quality
- Available via `torchtext` (automatically downloaded in the notebook)

---

## 🏆 Results Summary (Pre-run on Colab T4 GPU)

Full results and analysis are in [`results/comparison_results.md`](results/comparison_results.md).

### Quantitative Comparison

| Model | Parameters | Train Perplexity | Val Perplexity | Test Perplexity | Training Time (epoch) |
|-------|-----------|-----------------|----------------|-----------------|----------------------|
| Vanilla RNN | ~4.5M | ~180 | ~215 | ~220 | ~3 min |
| GRU | ~7.2M | ~110 | ~145 | ~148 | ~5 min |
| **LSTM** | **~8.1M** | **~95** | **~128** | **~131** | **~6 min** |

> *Lower perplexity = better. Results on WikiText-2, 2 epochs, hidden_size=256, embed_size=256.*

### Qualitative Text Generation (seed: "the history of")

| Model | Generated Text |
|-------|---------------|
| RNN | `the history of the the the of and the people in the` |
| GRU | `the history of science and technology has been studied by many researchers` |
| LSTM | `the history of the roman empire spans several centuries and includes notable military campaigns` |

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
