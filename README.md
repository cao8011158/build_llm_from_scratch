# Build LLM From Scratch

A minimal-from-scratch Transformer language model training stack:

**Tokenizer (BPE) → binarized dataset → TransformerLM (GQA + RoPE + RMSNorm + Online Softmax) → training with custom optimizers & schedulers**

This project implements a complete small-scale language model training
pipeline from scratch, including:

- Tokenizer training (BPE)
- Dataset binarization
- Transformer architecture
- Custom attention (Grouped-Query Attention + RoPE)
- Online softmax
- Optimizers & LR schedules
- Checkpointing & decoding

---

## 🚀 Features

### 🧩 Tokenizer

- BPE training
- Pretokenization pipeline
- Standalone or pipeline mode
- Located in: `src/llm_from_scratch/tokenizer/`

### 📦 Dataset

- Build `.bin` token datasets
- Efficient memory-mapped loading
- Random contiguous sequence sampling
- Located in: `src/llm_from_scratch/data/` and `serialization/`

### 🧠 Model

- TransformerLM
- Grouped-Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- Online Softmax (blockwise incremental softmax)
- Feedforward layers
- Located in: `src/llm_from_scratch/model/`

### 🏋️ Training

- YAML-based configuration system
- AdamW / SGD
- Cosine LR schedule with warmup
- Gradient clipping
- Checkpoint save & resume

---

## 📦 Project Structure

```
BUILD_LLM_FROM_SCRATCH/
│
├── configs/
│   ├── smoke.yaml
│   ├── test_overfitting.yaml
│   ├── training_config.yaml
│   └── training_config_lr*.yaml
│
├── src/llm_from_scratch/
│   ├── data/
│   ├── loss/
│   ├── model/
│   ├── optimizer/
│   ├── serialization/
│   ├── tokenizer/
│   ├── train.py
│   └── generate.py
│
└── pyproject.toml
```

---

# ⚙️ Setup

This project uses a `pyproject.toml`-based setup.

## Recommended (uv)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

Or using uv:

```bash
uv pip install -e .
```

After installation, all modules can be executed via:

```bash
python -m llm_from_scratch.<module>
```

---

# 🧩 Tokenizer

## 🟢 Option 1 — Train from Single Text File

```bash
python -m llm_from_scratch.tokenizer.train_bpe \
  --input-file path/to/text.txt \
  --out-dir artifacts/bpe \
  --vocab-size 32000
```

## 🔵 Option 2 — Train via Config (Pipeline Mode, Recommended)

```bash
python -m llm_from_scratch.tokenizer.train_bpe \
  --config configs/training_config.yaml
```

---

# 📦 Build Binary Dataset

After updating dataset paths inside `configs/training_config.yaml`, run:

```bash
python -m llm_from_scratch.serialization.build_bin_dataset
```

---

# 🧪 Debugging Modes

## Smoke Test

```bash
python -m llm_from_scratch.train \
  --config configs/smoke.yaml
```

## Overfitting Test

```bash
python -m llm_from_scratch.train \
  --config configs/test_overfitting.yaml
```

Expected behavior:

- Loss rapidly decreases
- Model memorizes a single batch
- Useful for debugging attention / gradients

---

# 🚀 Full Training

```bash
python -m llm_from_scratch.train \
  --config configs/training_config.yaml
```

---

# 🧠 Implementation Notes

## Grouped-Query Attention (GQA)

- Q uses `n_heads`
- K/V use `n_kv_heads`
- K/V are expanded to match Q heads
- RoPE applied to Q and K
- Causal masking
- Online Softmax computation

## Online Softmax

The attention block implements incremental softmax:

- Streaming/blockwise score computation
- Running max tracking
- Running normalization tracking
- Numerically stable
- Lower peak memory usage

## Dataset Format

- `.bin` files contain token IDs
- Stored as contiguous integer arrays
- Loaded via `numpy.memmap`
- High-throughput random sequence sampling

---

# 📊 Results

## Final Model Configuration

- Dataset: TinyStories (~2GB)
- Layers: 4
- Heads: 16
- Hidden size: 512
- Vocab size: 10,000
- Parameters: ~17M
- Batch size: 256
- Steps: 5,000
- Tokens seen: ~327M
- Precision: bf16
- Optimizer: AdamW
- LR schedule: Cosine + 250 warmup
- Gradient clipping: 1.0

---

## Learning Rate Sweep

| Learning Rate | Val Loss  | Val PPL  |
| ------------- | --------- | -------- |
| 2e-4          | 2.65      | 14.20    |
| 3e-4          | 2.31      | 10.04    |
| 6e-4          | 1.75      | 5.77     |
| 1e-3          | 1.55      | 4.73     |
| 3e-3          | **1.517** | **4.56** |

Best LR: **3e-3**

---

# 🔗 Pretrained Checkpoint & Tokenizer

## Model Checkpoint

Latest model:

https://drive.google.com/file/d/1xl1w4ITVL2dt5uzbhkoE1J6eDEBA0kVX/view?usp=sharing

Then update `configs/decode_tinystories.yaml`:

```yaml
checkpoint:
  path: /path/to/latest.pt
```

---

## Tokenizer Files

https://drive.google.com/drive/folders/1a-bsE5-vpHNI83qD6KuJGGvWsXOrq3Dv?usp=sharing

Then update `configs/decode_tinystories.yaml`:

```yaml
tokenizer:
  vocab_file: /path/to/vocab.json
  merges_file: /path/to/merges.txt
```

Then run:

```bash
python -m llm_from_scratch.generate \
  --config configs/decode_tinystories.yaml
```

## 📝 Example Output

```text
=== Decoding Settings ===
Prompt: Once upon a time
Max new tokens: 200
Temperature: 0.9
Top-p: 0.95
Device: cuda
Dtype: torch.bfloat16
=========================
Once upon a time, there was a little girl named Amy. She was a very obedient girl who always listened to her mom and dad. One day, her mom said, "Amy, it's time to get ready for the trip."
Amy went to the store and saw a big, modern toy. It was a toy car that could reverse. She asked her mom, "Can I have the toy car, please?" Her mom said, "Yes, you can have it, but be careful."
Amy was so happy. She played with the toy car all day. She knew that even in the big, expensive toy car, it could still have lots of fun. Amy and her mom went back home, and Amy had a great time with her new toy car.
```

---

# 📚 Acknowledgements

This project is based on:

Stanford CS336 Assignment 1 (Basics)

https://github.com/stanford-cs336/assignment1-basics/tree/main

Extended with:

- Grouped-Query Attention
- Online Softmax
