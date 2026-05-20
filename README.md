# GPT-2 Fine-Tuning — Next Word Prediction

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/GPT--2-Language_Model-412991?style=flat"/>
  <img src="https://img.shields.io/badge/Dataset-WikiText--2-4A90E2?style=flat"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat"/>
</p>

> Fine-tunes GPT-2 on WikiText-2 using the Hugging Face Trainer API, achieving **perplexity of 27.17** and **Top-3 accuracy of 56.97%** on the validation set.

---

## Problem Statement

Language models must learn contextual word relationships to generate fluent, coherent text. This project fine-tunes a pre-trained GPT-2 model on a curated text corpus to specialise it for next-word prediction — the core task behind autocomplete systems, writing assistants, and text generation APIs.

---

## Training Pipeline

```
WikiText-2 Dataset (Hugging Face Datasets)
            │
            ▼
  ┌──────────────────────────┐
  │  AutoTokenizer (GPT-2)   │  tokenise + block into fixed-length sequences
  └──────────────────────────┘
            │
            ▼
  ┌──────────────────────────┐
  │  DataCollatorForLM       │  causal LM collation (mlm=False)
  └──────────────────────────┘
            │
            ▼
  ┌──────────────────────────┐
  │  GPT2LMHeadModel         │  pre-trained weights, fine-tuned 2 epochs
  │  Hugging Face Trainer    │  lr=2e-5, batch=4, weight_decay=0.01
  └──────────────────────────┘
            │
            ▼
  ┌──────────────────────────┐
  │  Evaluation              │  perplexity + custom top-k accuracy
  └──────────────────────────┘
            │
            ▼
  ┌──────────────────────────┐
  │  Inference Pipeline      │  next-word generation on custom prompts
  └──────────────────────────┘
```

---

## Results

| Metric | Value |
|--------|-------|
| Perplexity | **27.17** |
| Top-3 Accuracy | **56.97%** |
| Training Epochs | 2 |
| Learning Rate | 2e-5 |
| Batch Size | 4 |
| Evaluation Strategy | Per epoch |

---

## Example Predictions

| Input Prompt | Model Prediction |
|-------------|-----------------|
| *Machine learning is* | a powerful technique used to analyze data. |
| *Artificial intelligence can* | transform industries by automating complex tasks. |
| *Deep learning models are* | capable of learning hierarchical representations. |

---

## Dataset

**WikiText-2** — a benchmark language modelling dataset from Hugging Face Datasets library.
- ~2M training tokens from verified Wikipedia articles
- Standard train/validation/test splits
- Loaded directly via `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language Model | GPT-2 (`GPT2LMHeadModel`) |
| Tokenizer | `AutoTokenizer` (GPT-2) |
| Training Framework | Hugging Face `Trainer` API |
| Data Collation | `DataCollatorForLanguageModeling` (causal LM) |
| Dataset | Hugging Face `datasets` — WikiText-2 |
| Deep Learning | PyTorch |
| Language | Python 3.10+ |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Cosmox999/gpt2-next-word-prediction.git
cd gpt2-next-word-prediction

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

Open [`notebooks/gpt2_next_word_prediction.ipynb`](notebooks/gpt2_next_word_prediction.ipynb) and run all cells:

1. Load and tokenise WikiText-2
2. Configure `DataCollatorForLanguageModeling` (causal, not masked)
3. Fine-tune GPT-2 with Hugging Face `Trainer`
4. Evaluate perplexity (`math.exp(eval_loss)`)
5. Run custom `compute_top_k_accuracy` on validation set
6. Generate predictions on custom prompts via pipeline

---

## Project Structure

```
├── notebooks/
│   └── gpt2_next_word_prediction.ipynb   # Full fine-tuning and evaluation pipeline
├── results/
│   ├── training_log.csv                  # Per-step loss and learning rate log
│   └── sample_predictions.txt            # Example next-word predictions
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Future Improvements

- **Larger model** — fine-tune GPT-2 Medium/Large for lower perplexity
- **Domain-specific corpus** — fine-tune on scientific or legal text
- **Beam search / top-p sampling** — richer generation strategies
- **Streamlit app** — interactive next-word prediction UI
- **ONNX export** — optimise for low-latency inference

---

## Author

**Ganesh Pandurang Sonawane**
Indian Institute of Technology Bombay

[![GitHub](https://img.shields.io/badge/GitHub-Cosmox999-181717?style=flat&logo=github)](https://github.com/Cosmox999)
