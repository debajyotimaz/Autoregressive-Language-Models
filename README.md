# 🧠 Autoregressive Language Models: RNN vs LSTM vs GRU

> **Tutorial for CS/NLP Lab | Monday, 23rd March 2026**  
> Prepared by **Suraj** and **Debajyoti** | Reviewed by Course Instructors  
> Theory grounded in: [Jurafsky & Martin, SLP3 — Chapter 13: RNNs and LSTMs](https://web.stanford.edu/~jurafsky/slp3/13.pdf)

---

## 📌 What This Tutorial Covers

This repository provides a **hands-on, end-to-end tutorial** on building **Autoregressive Language Models** using three recurrent architectures — **Vanilla RNN**, **LSTM**, and **GRU** — trained on a real Wikipedia text corpus using PyTorch.

By the end of this tutorial, participants will be able to:

1. Understand the theory of RNNs, LSTMs, and GRUs (following SLP3 Ch.13)
2. Implement all three architectures from scratch in PyTorch
3. Train autoregressive language models on Wikipedia data
4. Evaluate models using **Perplexity** and **BLEU score**
5. Generate text autoregressively from each trained model
6. Compare architectures **quantitatively** (metrics) and **qualitatively** (generated text)

---

## 🚀 Quick Start — Run in Google Colab

Click the badge below to open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/autoregressive-lm-tutorial/blob/main/AutoregressiveLM_Tutorial.ipynb)

> **Recommended:** Use a **GPU runtime** in Colab (`Runtime → Change runtime type → T4 GPU`) for faster training.

---

## 📁 Repository Structure

```
autoregressive-lm-tutorial/
│
├── AutoregressiveLM_Tutorial.ipynb   # ← MAIN Colab Notebook (start here)
│
├── README.md                          # This file
│
├── models/
│   ├── rnn_model.py                   # Vanilla RNN language model
│   ├── lstm_model.py                  # LSTM language model
│   └── gru_model.py                   # GRU language model
│
├── utils/
│   ├── dataset.py                     # Wikipedia data loader & tokenizer
│   ├── train.py                       # Training loop (shared for all models)
│   ├── evaluate.py                    # Perplexity & BLEU evaluation
│   └── generate.py                    # Autoregressive text generation
│
├── data/
│   └── README.md                      # Data download instructions
│
├── results/
│   └── comparison_results.md          # Pre-run results & analysis
│
└── requirements.txt                   # Python dependencies
```

---

## 📚 Theoretical Background

This tutorial is built directly on the theory from **SLP3 Chapter 13** by Jurafsky & Martin. Here's a quick map of concepts to notebook sections:

| SLP3 Section | Concept | Notebook Section |
|---|---|---|
| §13.1 | RNN architecture, forward inference, BPTT | Section 2 |
| §13.2 | RNNs as language models, teacher forcing | Section 3 |
| §13.2.3 | Weight tying | Section 4 |
| §13.3.3 | Autoregressive generation | Section 6 |
| §13.5 | Vanishing gradients → LSTM motivation | Section 2.3 |
| §13.6 | LSTM gates (input, forget, output) | Section 2.4 |
| §13.7 | GRU (simplified LSTM) | Section 2.5 |

### Key Equations (from SLP3 Ch.13)

**Vanilla RNN:**
```
hₜ = g(U·hₜ₋₁ + W·eₜ)
ŷₜ = softmax(Eᵀ·hₜ)          [weight tying]
```

**LSTM:**
```
fₜ = σ(Uf·hₜ₋₁ + Wf·xₜ)     [forget gate]
iₜ = σ(Ui·hₜ₋₁ + Wi·xₜ)     [input gate]
oₜ = σ(Uo·hₜ₋₁ + Wo·xₜ)     [output gate]
c̃ₜ = tanh(Ug·hₜ₋₁ + Wg·xₜ) [cell update]
cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ        [cell state]
hₜ = oₜ⊙tanh(cₜ)             [hidden state]
```

**GRU:**
```
zₜ = σ(Uz·hₜ₋₁ + Wz·xₜ)     [update gate]
rₜ = σ(Ur·hₜ₋₁ + Wr·xₜ)     [reset gate]
h̃ₜ = tanh(U·(rₜ⊙hₜ₋₁) + W·xₜ)
hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ
```

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

## 🗓 Tutorial Agenda (90 minutes)

| Time | Activity |
|------|---------|
| 0–15 min | Theory recap: RNN → LSTM → GRU (slides from SLP3 Ch.13) |
| 15–25 min | Walk through data loading & tokenization |
| 25–50 min | Implement & train all three models (run cells together) |
| 50–65 min | Evaluate: perplexity curves, comparison table |
| 65–80 min | Text generation demo & qualitative comparison |
| 80–90 min | Discussion: When to use which? Transformers vs RNNs? |

---

## 👥 Authors

- **Suraj** — [GitHub: @suraj]
- **Debajyoti** — [GitHub: @debajyoti]

*Tutorial prepared for the NLP/Deep Learning course, March 2026.*  
*Reviewed and approved by Course Instructors.*

---

## 📖 References

1. Jurafsky, D. & Martin, J.H. (2026). *Speech and Language Processing*, 3rd ed., Chapter 13. https://web.stanford.edu/~jurafsky/slp3/13.pdf
2. Elman, J.L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179–211.
3. Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
4. Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP 2014*.
5. Mikolov, T. et al. (2010). Recurrent neural network based language model. *Interspeech 2010*.
6. Merity, S. et al. (2017). Pointer Sentinel Mixture Models (WikiText-2 dataset). *ICLR 2017*.
