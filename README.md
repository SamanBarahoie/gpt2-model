
# GPT2Simple

A implementation of a GPT-2  Transformer using PyTorch, featuring Pre-LayerNorm and multi-head self-attention. Designed for learning, experimentation, and educational purposes.

---

## Features

- Pre-LayerNorm architecture (GPT-2 )
- Multi-head self-attention with causal masking
- Token and positional embeddings
- Feed-forward layers with GELU activation
- Configurable model using Python dataclass
- Weight tying between input and output embeddings
- Clean and well-structured PyTorch code

---

## Requirements

- Python 3.8+
- PyTorch >= 1.10

Install dependencies:

```bash
pip install torch
```

---

## Usage

The model can be instantiated and tested with a random input batch:

```bash
python gpt2_simple.py
```

Expected output:

```
Logits shape: torch.Size([2, 16, 50257])
```

---

## File Overview

- `gpt2_simple.py`: Core model definition and execution example.

---

## Model Overview

Architecture:

```
Input IDs
   │
Token + Positional Embeddings
   │
 [Transformer Block × N]
   │
LayerNorm
   │
Linear (Tied to token embeddings)
   │
Logits
```

Each transformer block includes:

- LayerNorm
- Multi-head self-attention
- Feed-forward MLP
- Residual connections

---

## Configuration

You can define all model hyperparameters in one place using the `Config` dataclass:

```python
config = Config(
    vocab_size=50257,
    d_model=768,
    num_heads=12,
    num_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout_prob=0.1
)
```

---

## License

This project is provided for educational purposes.
```
