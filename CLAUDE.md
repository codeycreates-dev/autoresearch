# Autoresearch

Autonomous AI research agent that experiments with GPT model training. Modifies `train.py`, runs 5-minute experiments, keeps improvements, discards regressions.

## Key Files
- `train.py` — GPT model architecture, optimizer, training loop (the AI agent modifies this)
- `prepare.py` — Data download, tokenizer training, dataloader utilities (fixed, don't modify)
- `program.md` — Research instructions for the AI agent (human edits this)
- `GUIDE.ipynb` — Interactive beginner guide explaining every concept in the codebase

## Tech Stack
Python, PyTorch, single NVIDIA GPU, `uv` package manager

## Run
```bash
uv run prepare.py   # download data & train tokenizer (first time only)
uv run train.py     # train the model
```

## Key Metric
`val_bpb` (validation bits per byte) — lower is better.

## Architecture
GPT-style transformer: 12 layers, 6 heads, 768 embedding dim, ReLU² activation, Flash Attention 3, RoPE, sliding window attention. Trained with Muon + AdamW optimizers.
