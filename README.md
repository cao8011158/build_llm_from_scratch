# Build LLM From Scratch

A minimal-from-scratch Transformer language model training stack:

**Tokenizer (BPE) → binarized token dataset → TransformerLM (GQA + RoPE,
RMSNorm, FFN) → training with custom optimizers & schedulers**

This project implements a full small-scale language model training
pipeline from scratch, including tokenizer training, dataset
binarization, Transformer architecture, optimization, and checkpointing.

---

## 🚀 Features

- **Tokenizer**
  - BPE training
  - Pretokenization pipeline\
  - Located in: `src/llm_from_scratch/tokenizer/`
- **Dataset**
  - Build `.bin` token dataset
  - Efficient batch loading
  - Located in: `src/llm_from_scratch/data/` and `serialization/`
- **Model**
  - TransformerLM
  - Grouped-Query Attention (GQA)
  - RoPE positional encoding
  - RMSNorm
  - Feedforward layers
  - Located in: `src/llm_from_scratch/model/`
- **Training**
  - YAML-based config system
  - AdamW / SGD optimizers
  - Learning rate schedules
  - Gradient clipping
  - Checkpointing support

---

## 📦 Project Structure

    BUILD_LLM_FROM_SCRATCH/
      configs/
        smoke.yaml
        test_overfitting.yaml
        training_config.yaml
        training_config_lr*.yaml
      src/llm_from_scratch/
        data/
          get_batch.py
        loss/
          cross_entropy.py
        model/
          embedding.py
          gqa_self_attention.py
          linear.py
          positional_embedding.py
          RMSNorm.py
          transformer_block.py
          transformer_lm.py
          ops/
        optimizer/
          adamw.py
          sgd.py
          schedule.py
          gradient_clipping.py
        serialization/
          build_bin_dataset.py
          checkpointing.py
        tokenizer/
          bpe_tokenizer.py
          pretokenization.py
          train_bpe.py
          pretokenization_example.py

---

## ⚙️ Quickstart

### 1️⃣ Setup

This project uses `pyproject.toml` for dependency management.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

### 2️⃣ Train Tokenizer

### 🟢 Option 1 --- Train from a Single Text File (No Config Required)

```bash
python -m llm_from_scratch.tokenizer.train_bpe \
  --input-file path/to/text_corpus.txt \
  --out-dir artifacts/bpe \
  --vocab-size 32000
```

### 🔵 Option 2 --- Train Using Dataset + Config (Pipeline Mode)

Use this mode when working with the full LLM training pipeline
(TinyStories, OWT, etc.).

```bash
python -m llm_from_scratch.tokenizer.train_bpe \
  --dataset tinystories \
  --config configs/training_config.yaml \
  --out-dir artifacts/bpe \
  --vocab-size 32000
```

### 3️⃣ Build Bin Dataset

Update paths in `configs/training_config.yaml`, then run:

```bash
python -m src.llm_from_scratch.serialization.build_bin_dataset
```

### 4️⃣ Run Smoke Test

```bash
python train.py --config configs/smoke.yaml
```

### 5️⃣ Start Training

```bash
python train.py --config configs/training_config.yaml
```

---

## 🧠 Implementation Notes

### GQA Attention

- Q uses `n_heads`
- K/V use `n_kv_heads`
- K/V are expanded to match Q heads
- RoPE is applied to Q and K before attention computation
- Online Softmax is used for numerically stable and memory-efficient attention computation

### Online Softmax

The attention block implements an incremental (online) softmax algorithm, which:

- Computes attention scores in a streaming/blockwise manner
- Maintains running max and normalization terms
- Improves numerical stability
- Reduces peak memory usage compared to naive full softmax

### Dataset Format

- `.bin` files contain token IDs
- Stored as contiguous integer arrays
- Loaded efficiently using memory mapping (`numpy.memmap`)
- Random contiguous sequences sampled during training

## 📊 Results

### Final Model Configuration

- Dataset: TinyStories (~2GB)
- Transformer Layers: 4
- Attention Heads: 16
- Hidden Dimension: 512
- Vocabulary Size: 10,000
- Total Parameters: ~17M
- Batch Size: 256
- Training Steps: 5,000
- Total Tokens Seen: ~327M
- Precision: bf16
- Optimizer: AdamW
- LR Schedule: Cosine decay with 250-step warmup
- Gradient Clipping: 1.0

---

### Learning Rate Sweep

| Learning Rate | Val Loss  | Val Perplexity |
| ------------- | --------- | -------------- |
| 2e-4          | 2.65      | 14.20          |
| 3e-4          | 2.31      | 10.04          |
| 6e-4          | 1.75      | 5.77           |
| 1e-3          | 1.55      | 4.73           |
| 3e-3          | **1.517** | **4.56**       |

---

### Best Model

- Best Learning Rate: **3e-3**
- Final Validation Loss: **1.517**
- Final Validation Perplexity: **4.56**

Increasing the learning rate improved convergence within the tested range.
Due to computational constraints, higher learning rates were not explored further.

## 📚 Acknowledgements

-This project is based on the Stanford CS336 Assignment 1 (Basics) starter code and structure, and has been extended with additional components

-Source: https://github.com/stanford-cs336/assignment1-basics/tree/main

```

```
