"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py

--- HIGH-LEVEL OVERVIEW ---

This script trains a GPT-style language model from scratch on a single GPU.

WHAT IS A LANGUAGE MODEL?
A language model predicts the next word (token) given all the previous words.
Given "The cat sat on the", it should predict "mat" (or similar). By learning
to do this well on billions of tokens of text, the model learns grammar,
facts, reasoning patterns, and more.

WHAT IS GPT?
GPT (Generative Pre-trained Transformer) is an architecture for language models.
It's built from stacked "transformer blocks", each containing:
1. Self-attention: lets each token look at all previous tokens to gather context
2. Feed-forward network (MLP): processes each token independently to add capacity

This file contains:
- The GPT model architecture (embeddings, attention, MLP, output head)
- A custom optimizer (Muon + AdamW) for efficient training
- The training loop with learning rate scheduling
- Hyperparameters you can tune

The AI agent modifies THIS file to try different architectures, hyperparameters,
and training strategies to achieve the lowest val_bpb (validation bits per byte).
"""

import os
# Tell PyTorch to use expandable memory segments — reduces memory fragmentation
# on the GPU, which prevents out-of-memory errors from fragmented free space.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Suppress HuggingFace download progress bars (would clutter training output)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn          # Neural network building blocks (layers, etc.)
import torch.nn.functional as F # Functional operations (activation functions, loss, etc.)

# "kernels" is a package manager for GPU compute kernels (low-level GPU functions).
# Flash Attention 3 is a highly optimized implementation of the attention mechanism
# that's much faster and more memory-efficient than the naive implementation.
# It computes the same result but uses clever tiling to avoid materializing the
# full attention matrix in GPU memory.
from kernels import get_kernel
cap = torch.cuda.get_device_capability()  # e.g., (9, 0) for H100, (8, 9) for RTX 4090
# Use the official FA3 kernel for Hopper GPUs (H100), otherwise use a community port
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

# Import fixed constants and utilities from prepare.py
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Configuration for the GPT model architecture.

    These parameters define the shape and size of the model. Together they
    determine how many parameters (learnable numbers) the model has, which
    directly affects its capacity to learn patterns.
    """
    sequence_len: int = 2048    # Max number of tokens the model can see at once (context window)
    vocab_size: int = 32768     # Number of unique tokens the model knows (overridden at runtime)
    n_layer: int = 12           # Number of transformer blocks stacked on top of each other (depth)
    n_head: int = 6             # Number of attention heads (parallel attention computations)
    n_kv_head: int = 6          # Number of key/value heads (can be < n_head for "grouped query attention")
    n_embd: int = 768           # Embedding dimension — the size of each token's vector representation
    window_pattern: str = "SSSL" # Sliding window attention pattern (S=short window, L=long/full window)


def norm(x):
    """RMS Normalization — a technique to stabilize training.

    Neural networks can have their internal values grow very large or very small
    as data flows through many layers. Normalization keeps values in a reasonable
    range. RMS norm divides each vector by its root-mean-square value.

    Compared to the more common LayerNorm, RMSNorm skips the mean-subtraction
    step, which makes it slightly faster with similar effectiveness.
    """
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if this layer should have a Value Embedding.

    Value embeddings are applied to alternating layers (every other layer),
    with the last layer always included. This saves memory and compute
    compared to having value embeddings on every layer.
    """
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """Apply Rotary Position Embeddings (RoPE) to queries or keys.

    THE PROBLEM: Self-attention is "position-blind" by default — it doesn't know
    if a token is at position 1 or position 100. But word order matters!
    "Dog bites man" means something different from "Man bites dog."

    THE SOLUTION: RoPE encodes position information by rotating each token's
    query/key vectors by an angle that depends on the token's position. Tokens
    at different positions get rotated differently, so when attention computes
    the dot product between a query and a key, the result naturally depends on
    their relative positions.

    HOW IT WORKS (simplified):
    - Split each vector into pairs of dimensions: (x1, x2)
    - Rotate each pair by an angle θ that depends on the position:
      y1 = x1 * cos(θ) + x2 * sin(θ)
      y2 = -x1 * sin(θ) + x2 * cos(θ)
    - Different dimension pairs use different frequencies, so the model can
      encode both fine-grained (nearby) and coarse (distant) position info.
    """
    assert x.ndim == 4  # Shape: [batch, sequence_length, num_heads, head_dim]
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """The self-attention mechanism — the core of the transformer.

    WHAT ATTENTION DOES:
    For each token in the sequence, attention answers: "which other tokens
    should I pay attention to, and how much?" It computes this by:

    1. PROJECT each token into three vectors:
       - Query (Q): "what am I looking for?"
       - Key (K): "what do I contain?"
       - Value (V): "what information do I provide?"

    2. SCORE: compute dot product between each Query and all Keys.
       High score = these tokens are relevant to each other.

    3. ATTEND: use scores as weights to compute a weighted sum of Values.
       Each token's output is a blend of all relevant tokens' information.

    "Causal" means each token can only attend to tokens BEFORE it (not future
    tokens), because during text generation, future tokens don't exist yet.

    MULTI-HEAD ATTENTION:
    Instead of one big attention computation, we split into multiple "heads"
    (e.g., 6). Each head can focus on different types of relationships
    (one head might focus on syntax, another on semantics, etc.).

    VALUE EMBEDDING (ve):
    This is a ResFormer technique. In addition to the normal value projection
    from the hidden state, we also look up a separate "value embedding" directly
    from the input token IDs. This gives the model a shortcut to access raw
    token information in deeper layers, which helps training stability.
    The gate controls how much of this direct signal to mix in.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head  # Size of each attention head's vectors

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # Linear projections: transform the embedding into Q, K, V vectors
        # "Linear" = a matrix multiplication (the most fundamental neural net operation)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)     # Query projection
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # Key projection
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # Value projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)  # Output projection

        # Value Embedding gate: controls how much of the direct token embedding
        # to mix into the value vectors. Uses a small subset of input channels
        # (32) to save compute. Only present on alternating layers.
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        """
        Args:
            x: Input tensor [batch, seq_len, embedding_dim]
            ve: Value embedding from token IDs [batch, seq_len, kv_dim], or None
            cos_sin: Precomputed (cos, sin) for rotary embeddings
            window_size: Tuple for sliding window attention (window_len, 0)
        """
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Project input into queries, keys, and values, then reshape for multi-head
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer technique): mix in the value embedding with
        # an input-dependent gate. The gate uses sigmoid (output 0 to 1), scaled
        # by 2, so the default (sigmoid(0)*2 = 1.0) is neutral — it adds the
        # value embedding with weight 1.0 by default.
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            # Gate depends on the first 32 channels of x — one gate value per KV head
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve  # Mix in value embedding

        # Apply rotary position embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # Normalize Q and K before attention (QK-norm). This prevents attention
        # scores from becoming too large, which stabilizes training at scale.
        q, k = norm(q), norm(k)

        # Flash Attention 3: the actual attention computation.
        # causal=True means each token can only attend to itself and earlier tokens.
        # window_size limits how far back each token can look (sliding window).
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)  # Reshape back: merge all heads

        # Final linear projection to mix information across heads
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Feed-Forward Network (MLP) — the other half of each transformer block.

    While attention lets tokens communicate with each other, the MLP processes
    each token independently. It expands the representation to 4x the embedding
    dimension, applies a nonlinearity, then projects back down.

    The nonlinearity here is ReluSquared (ReLU(x)^2):
    - ReLU(x) = max(0, x) — zeroes out negative values
    - Squaring amplifies large positive values and keeps sparsity from ReLU
    This creates a "sparse" activation pattern where many neurons are zero,
    which has been shown to improve quality in some settings.
    """
    def __init__(self, config):
        super().__init__()
        # "up projection": expand from n_embd to 4*n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # "down projection": compress back from 4*n_embd to n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)           # Expand: [B, T, n_embd] -> [B, T, 4*n_embd]
        x = F.relu(x).square()     # ReluSquared activation
        x = self.c_proj(x)         # Compress: [B, T, 4*n_embd] -> [B, T, n_embd]
        return x


class Block(nn.Module):
    """A single transformer block = Attention + MLP, both with residual connections.

    RESIDUAL CONNECTIONS:
    Instead of x = attention(x), we do x = x + attention(norm(x)).
    The "+x" is the residual connection — it lets information flow directly
    through the block without being forced through the attention/MLP computation.
    This is critical for training deep networks: without residuals, gradients
    vanish as they flow backward through many layers, making learning impossible.

    The norm() before each sub-layer (attention, MLP) stabilizes the input
    distribution, which helps training converge faster and more reliably.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)  # Attention + residual
        x = x + self.mlp(norm(x))                              # MLP + residual
        return x


class GPT(nn.Module):
    """The full GPT language model.

    ARCHITECTURE OVERVIEW:
    1. Token Embedding (wte): Converts token IDs to dense vectors.
       Each token ID (e.g., 1234) gets mapped to a learned vector of size n_embd.
       This is the model's "vocabulary" — a lookup table of vector representations.

    2. Stack of Transformer Blocks: The core of the model. Each block refines
       the token representations through attention (cross-token communication)
       and MLP (per-token processing). With 8 blocks, information flows through
       8 rounds of refinement.

    3. Language Model Head (lm_head): Projects the final representations back
       to vocabulary size to produce logits (scores) for each possible next token.
       Higher logit = model thinks that token is more likely to come next.

    SPECIAL FEATURES IN THIS IMPLEMENTATION:

    - Residual Lambdas (resid_lambdas, x0_lambdas): Per-layer learnable scalars
      that control the residual connection. Instead of x = x + block(x), we do
      x = λ_resid * x + λ_x0 * x0 + block(x), where x0 is the initial embedding.
      This gives the model a "highway" back to the original token embeddings at
      every layer, helping deep networks train better.

    - Value Embeddings: Separate embedding tables that provide a direct path from
      input token IDs to the value vectors in attention. This is the ResFormer
      technique — it helps because deeper layers can directly access token info
      without it being "washed out" by many layers of processing.

    - Sliding Window Attention: Some layers use a shorter attention window
      (half the sequence length) instead of attending to all previous tokens.
      Pattern "SSSL" = Short, Short, Short, Long, repeating. This saves compute
      because most relevant context is nearby, and the occasional "Long" layer
      can still capture distant dependencies.

    - Logit Softcapping: The output logits are passed through tanh scaled by 15,
      which limits them to the range [-15, +15]. This prevents extreme predictions
      that could cause numerical instability during training.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Compute the sliding window sizes for each layer based on the pattern
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            # Token embedding: lookup table mapping token_id -> vector of size n_embd
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            # Stack of transformer blocks
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        # Language model head: projects from embedding space -> vocabulary space
        # to produce next-token predictions
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-layer scalars for the enhanced residual connection:
        # x = resid_lambda[i] * x + x0_lambda[i] * x0
        # where x is the running hidden state and x0 is the initial embedding
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # Initialized to 1.0 (normal residual)
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # Initialized to 0.0 (no x0 shortcut initially)

        # Value embeddings: separate embedding tables for the value residual technique.
        # Only created for alternating layers (to save memory).
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        # Precompute rotary embeddings for all possible positions.
        # We precompute 10x the sequence length to have headroom.
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # register_buffer = store as part of the model but not a learnable parameter
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()  # No gradients needed during initialization
    def init_weights(self):
        """Initialize all model weights with carefully chosen random values.

        Good weight initialization is crucial for training deep networks. Bad init
        can cause gradients to explode or vanish from the start, making training
        fail entirely. The key principles here:

        - Embedding: normal distribution with std=1.0 (standard)
        - Output head (lm_head): very small init (std=0.001) so initial predictions
          are near-uniform across the vocabulary (the model starts "unsure")
        - Attention/MLP weights: uniform in [-s, s] where s = sqrt(3) / sqrt(n_embd).
          This keeps the variance of activations stable across layers.
        - Output projections (c_proj): initialized to zero. Combined with the residual
          connection, this means each block initially acts as an identity function
          (input passes through unchanged). Training gradually "turns on" each block.
        - VE gates: initialized to zero, so sigmoid(0)=0.5, scaled by 2 = 1.0.
          This means value embeddings start with neutral (1x) contribution.
        """
        # Embedding and unembedding (output head)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # sqrt(3) / sqrt(n_embd) — keeps variance stable
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)   # Zero init for residual
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)    # Zero init for residual
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # Start with standard residual (1*x)
        self.x0_lambdas.fill_(0.1)      # Small initial contribution from x0
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bfloat16 for mixed-precision training
        # (bfloat16 uses 16 bits instead of 32, halving memory and doubling throughput)
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """Precompute cos and sin values for Rotary Position Embeddings (RoPE).

        RoPE uses sinusoidal functions at different frequencies to encode position.
        Lower frequency dimensions capture long-range position info, higher frequency
        dimensions capture fine-grained nearby position info. The base (10000) controls
        the frequency distribution.

        Returns cos and sin tensors shaped [1, seq_len, 1, head_dim//2] — the extra
        dimensions of size 1 are for broadcasting across batch and head dimensions.
        """
        if device is None:
            device = self.transformer.wte.weight.device
        # Create inverse frequencies: each pair of dimensions gets a different frequency
        # Lower dimensions = higher frequency, higher dimensions = lower frequency
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # Position indices
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # Outer product: position × frequency = angles for each (position, dimension) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Convert to bfloat16 for speed, reshape for broadcasting
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # [1, seq_len, 1, head_dim//2]
        return cos, sin

    def _compute_window_sizes(self, config):
        """Compute sliding window sizes for each transformer layer.

        SLIDING WINDOW ATTENTION:
        Full attention lets each token attend to ALL previous tokens — O(n²) cost.
        Sliding window attention limits each token to only the nearest W tokens.
        This is much cheaper and works well because most useful context is nearby.

        The pattern "SSSL" repeats across layers:
        - S (Short): attend to the nearest half of the sequence (e.g., 1024 tokens)
        - L (Long): attend to the full sequence (2048 tokens)

        The last layer always uses Long (full attention) to ensure the model can
        still capture any long-range dependencies.
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len     # Full context length
        short_window = long_window // 2       # Half context length
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            # Cycle through the pattern: "SSSL" -> S, S, S, L, S, S, S, L, ...
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Override: last layer always uses full attention
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimate FLOPs (floating-point operations) per token for forward + backward pass.

        FLOPs tells you how much computation the model needs. The rule of thumb for
        transformers is: 6 * num_params FLOPs per token (2 for forward, 4 for backward
        with gradient computation). Attention has additional FLOPs proportional to
        sequence length × head dimension × number of heads.

        This is used to compute MFU (Model FLOPs Utilization) — what fraction of the
        GPU's theoretical peak performance we're actually achieving. Good MFU means
        we're using the hardware efficiently.
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude parameters that don't contribute to the standard "6N" estimate
        # (embeddings, value embeddings, and scalar parameters)
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head  # head_dim
        t = self.config.sequence_len
        # Attention FLOPs depend on the window size (shorter window = less compute)
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        """Count parameters by category — useful for understanding model composition."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        """Configure the optimizer with different learning rates for different parameter types.

        WHY DIFFERENT LEARNING RATES?
        Different types of parameters learn best at different speeds:
        - Embeddings (token lookup tables): high LR because they're sparse (only a few
          tokens get updated each step)
        - Matrix parameters (attention/MLP weights): medium LR, use Muon optimizer
        - Output head (lm_head): low LR because it maps to the full vocabulary and
          small changes here have outsized effects on predictions
        - Scalars (resid_lambdas, x0_lambdas): these are single numbers per layer,
          need careful tuning

        Learning rates are also scaled by 1/sqrt(model_dim/768) — larger models need
        smaller learning rates to train stably. This is tuned assuming dim=768 as baseline.

        TWO OPTIMIZERS IN ONE:
        - AdamW: the industry standard optimizer. Maintains running averages of
          gradients (momentum) and squared gradients (adaptive learning rate).
          Used for embeddings, output head, and scalars.
        - Muon: a newer optimizer designed for matrix-shaped parameters. Uses
          "orthogonalization" to keep weight matrices well-conditioned, plus
          Nesterov momentum. Generally trains faster than Adam for these params.
        """
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # Sanity check: make sure we accounted for ALL parameters
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale learning rate inversely with sqrt of model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            # Output head: very small LR because it maps directly to predictions
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # Token embeddings: high LR because they're sparse
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # Value embeddings: same treatment as token embeddings
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # Residual lambdas: very small LR (these are sensitive scalars)
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # x0 lambdas: higher LR with different beta1 (0.96 for more momentum)
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Matrix parameters (attention/MLP weights) use Muon optimizer.
        # Group by shape because Muon stacks all same-shape params into a batch.
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        # Store initial LR for each group so the scheduler can compute LR = initial_lr * multiplier
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        """Forward pass: token IDs in, logits (or loss) out.

        THE FORWARD PASS (what happens when the model processes text):

        1. EMBED: Look up each token ID in the embedding table to get a vector.
           [batch, seq_len] of ints -> [batch, seq_len, n_embd] of floats

        2. NORMALIZE: Apply RMS normalization to the embeddings.

        3. SAVE x0: Store the initial embeddings for the x0 residual shortcut.

        4. TRANSFORMER BLOCKS: Pass through each block sequentially. Before each
           block, apply the enhanced residual: x = λ_resid * x + λ_x0 * x0
           This blends the running hidden state with the original embeddings.

        5. FINAL NORM: Normalize the output of the last block.

        6. SOFTCAP + LOGITS: Project to vocabulary size and apply tanh softcapping.
           Softcap of 15 means logits are limited to [-15, +15], preventing extreme
           confidence that could cause numerical issues in the loss computation.

        7. LOSS (if targets provided): Compute cross-entropy loss — how "wrong"
           the model's predictions are. Lower loss = better predictions.
        """
        B, T = idx.size()  # Batch size, sequence length
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]  # Slice rotary embeddings to seq length

        # Step 1-3: Embed tokens, normalize, save initial state
        x = self.transformer.wte(idx)  # [B, T] -> [B, T, n_embd]
        x = norm(x)
        x0 = x  # Save for x0 residual shortcut

        # Step 4: Pass through all transformer blocks
        for i, block in enumerate(self.transformer.h):
            # Enhanced residual: blend running state with original embeddings
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # Look up value embedding for this layer (if it has one)
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])

        # Step 5-6: Final normalization, project to vocab, and softcap
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)        # [B, T, n_embd] -> [B, T, vocab_size]
        logits = logits.float()          # Cast to float32 for numerical stability
        logits = softcap * torch.tanh(logits / softcap)  # Softcap: limit to [-15, +15]

        # Step 7: Compute loss if targets are provided
        if targets is not None:
            # Cross-entropy loss: measures how far the model's predictions are from
            # the actual next tokens. Lower = better.
            # view(-1) flattens to 1D: [B*T, vocab_size] vs [B*T]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

# Precomputed polynomial coefficients for the "Polar Express" orthogonalization.
# These approximate the matrix polar decomposition (finding the nearest orthogonal
# matrix) using a few cheap matrix multiplications instead of expensive SVD.
# Each tuple (a, b, c) defines one iteration: X_new = a*X + X @ (b*A + c*A²)
# where A = X^T @ X. 5 iterations gives a good approximation.
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)  # JIT-compile for GPU efficiency
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    """One step of AdamW optimizer, fused into a single GPU kernel for speed.

    AdamW is the standard optimizer for deep learning. It maintains two running averages:
    - exp_avg (first moment): exponential moving average of gradients (momentum).
      This smooths out noisy gradients so the optimizer moves in a consistent direction.
    - exp_avg_sq (second moment): exponential moving average of squared gradients.
      This adapts the learning rate per-parameter: parameters with large gradients
      get smaller effective LR, and vice versa.

    The "W" in AdamW means "decoupled weight decay" — it shrinks weights directly
    (p *= 1 - lr*wd) rather than adding a penalty to the loss. This is better
    for regularization.
    """
    # Weight decay: slightly shrink all weights toward zero (regularization)
    p.mul_(1 - lr_t * wd_t)
    # Update first moment (gradient momentum): blend old average with new gradient
    exp_avg.lerp_(grad, 1 - beta1_t)
    # Update second moment (squared gradient): blend old average with new gradient²
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias correction: the running averages start at zero, so early estimates are
    # biased toward zero. These corrections compensate for that.
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Adaptive step: divide momentum by sqrt of second moment (+ epsilon for stability)
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    # Apply the update to the parameter
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    """One step of the Muon optimizer — a specialized optimizer for matrix parameters.

    Muon is designed specifically for 2D weight matrices in neural networks. The key
    idea: instead of just following the gradient direction, Muon first "orthogonalizes"
    the gradient update. This keeps weight matrices well-conditioned (not collapsing
    into low-rank or poorly scaled states), which empirically leads to faster training.

    THE STEPS:

    1. NESTEROV MOMENTUM: A variant of momentum that "looks ahead" — it evaluates
       the gradient at a predicted future position, which gives better convergence
       than standard momentum.

    2. POLAR EXPRESS ORTHOGONALIZATION: Finds the nearest orthogonal matrix to the
       gradient. An orthogonal matrix has all singular values = 1, meaning the update
       preserves the scale of the weight matrix. This is computed using a fast
       polynomial approximation (5 iterations) instead of expensive SVD.

    3. NORMUON VARIANCE REDUCTION: Normalizes the update using a running average of
       variance per row/column. This acts like Adam's adaptive learning rate but
       tailored for the orthogonalized update. It prevents any single row or column
       from dominating the update.

    4. CAUTIOUS WEIGHT DECAY: Only applies weight decay where the gradient and parameter
       agree in sign (both positive or both negative). This prevents weight decay from
       fighting against the gradient direction, which can slow down learning.
    """
    # Step 1: Nesterov momentum
    # Standard momentum: buf = momentum * buf + grad
    # Nesterov twist: use buf + momentum * (new_buf - old_buf), which looks ahead
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Step 2: Polar Express orthogonalization
    # This iteratively projects the gradient toward the nearest orthogonal matrix.
    # For tall matrices (rows > cols), we compute X^T @ X.
    # For wide matrices (cols > rows), we compute X @ X^T.
    # The polynomial iteration refines X until it's approximately orthogonal.
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)  # Normalize for stability
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X          # Gram matrix
            B = b * A + c * (A @ A)  # Polynomial of the Gram matrix
            X = a * X + X @ B       # Update X
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT          # Gram matrix (transposed version)
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Step 3: NorMuon variance reduction
    # Compute per-row (or per-column) variance of the update, maintain a running
    # average, and use it to normalize. This makes the effective step size uniform
    # across all rows/columns of the weight matrix.
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Step 4: Cautious weight decay + parameter update
    # "Cautious": only decay where gradient and param agree in sign.
    # This prevents weight decay from opposing the gradient direction.
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0  # True where gradient and param have same sign
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for everything else.

    This is a "meta-optimizer" that dispatches to the right algorithm based on
    the parameter type. The idea is that matrix-shaped parameters (attention
    projections, MLP layers) benefit from Muon's orthogonalization, while
    1D parameters (embeddings, scalars, biases) work fine with standard AdamW.

    Implementation detail: scalar tensors (like learning rate, momentum) are
    stored as 0-dimensional CPU tensors rather than Python floats. This prevents
    torch.compile from recompiling the fused kernels every time a value changes.
    """

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors: these are hyperparameters passed to the compiled kernels.
        # Using tensors instead of floats avoids torch.compile recompilation when
        # values change between steps (e.g., learning rate schedule).
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        """Run one AdamW step for all parameters in this group."""
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            # Initialize optimizer state on first step
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)     # First moment (momentum)
                state['exp_avg_sq'] = torch.zeros_like(p)   # Second moment (adaptive LR)
            state['step'] += 1
            # Fill scalar tensors with current values (avoids recompilation)
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        """Run one Muon step for all parameters in this group.

        Muon processes all same-shape parameters as a batch (stacked into a 3D tensor)
        for efficiency. This is why parameters are grouped by shape in setup_optimizer.
        """
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        # Initialize momentum buffers on first step
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            # Second momentum is per-row or per-column (whichever is smaller)
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        # Reduce along the smaller dimension (row or column)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        # Stack all same-shape params and grads into batched tensors
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        # Fill scalar tensors
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        # Scale LR by sqrt(aspect_ratio) for non-square matrices
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        # Run the fused Muon step
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        # Copy updated stacked params back into the individual parameter tensors
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()  # No gradient tracking during optimizer step itself
    def step(self):
        """Run one optimization step: update all parameters based on their gradients."""
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------
# These are the knobs the AI agent turns to find the best model configuration.
# All of them affect the final val_bpb metric.

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO (controls width/depth ratio)
HEAD_DIM = 128          # Dimension of each attention head (128 is standard for modern models)
WINDOW_PATTERN = "SSSL" # Sliding window pattern: S=half context, L=full context per layer

# Optimization — learning rates for different parameter types
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step. Larger batch = more stable gradients
                         # but fewer update steps in the fixed time budget
EMBEDDING_LR = 0.6      # Learning rate for token embeddings (Adam) — high because sparse
UNEMBEDDING_LR = 0.004  # Learning rate for lm_head (Adam) — low because sensitive
MATRIX_LR = 0.04        # Learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # Learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # Cautious weight decay for Muon — regularization strength
ADAM_BETAS = (0.8, 0.95) # Adam momentum parameters: (beta1, beta2)
                          # beta1=0.8: faster momentum decay (more responsive to recent gradients)
                          # beta2=0.95: moderately fast second moment decay
WARMUP_RATIO = 0.0      # Fraction of time budget for LR warmup (0.0 = no warmup)
WARMDOWN_RATIO = 0.5    # Fraction of time budget for LR cooldown (last 50% of training)
FINAL_LR_FRAC = 0.0     # Final LR as fraction of initial (0.0 = LR reaches zero at end)

# Model size
DEPTH = 8               # Number of transformer layers. model_dim = 8 * 64 = 512
DEVICE_BATCH_SIZE = 128  # How many sequences to process per forward/backward pass.
                         # Reduce if you run out of GPU memory (OOM).

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
# Set random seeds for reproducibility — same seed = same initial weights and data order
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Allow TF32 for matmuls — a NVIDIA hardware feature that trades tiny precision
# for ~3x speed on matrix multiplications. The precision loss is negligible for training.
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
# Mixed precision: run most computations in bfloat16 (16-bit) instead of float32 (32-bit).
# This halves memory usage and roughly doubles throughput on modern GPUs, with minimal
# quality impact. bfloat16 has the same exponent range as float32 (so no overflow issues)
# but fewer mantissa bits (less precision).
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
# Theoretical peak FLOPS for H100 in bfloat16 — used to compute MFU (efficiency metric)
H100_BF16_PEAK_FLOPS = 989.5e12  # ~990 trillion operations per second

# Load the tokenizer that was trained by prepare.py
tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    """Build model config from depth. Model width = depth * ASPECT_RATIO, rounded up
    to be divisible by HEAD_DIM (so we get a whole number of attention heads)."""
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM  # Round up to multiple of HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

# Create model on "meta" device first (no actual memory allocated), then move to GPU
# and initialize weights. This two-step process is more memory-efficient than creating
# directly on GPU because it avoids allocating default random weights that get
# immediately overwritten by init_weights().
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)  # Allocate real GPU memory (uninitialized)
model.init_weights()           # Fill with properly initialized values

# Print parameter counts broken down by component
param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# GRADIENT ACCUMULATION: If TOTAL_BATCH_SIZE is larger than what fits in GPU memory
# at once (DEVICE_BATCH_SIZE * MAX_SEQ_LEN tokens), we split the batch into multiple
# "micro-steps". Each micro-step processes DEVICE_BATCH_SIZE sequences and accumulates
# gradients. After all micro-steps, we do one optimizer update.
# Example: TOTAL_BATCH_SIZE = 524288, DEVICE_BATCH_SIZE = 128, MAX_SEQ_LEN = 2048
#   tokens_per_fwdbwd = 128 * 2048 = 262144
#   grad_accum_steps = 524288 / 262144 = 2 micro-steps per optimizer step
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# Set up the optimizer with per-parameter-type learning rates
optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# torch.compile: JIT-compiles the model into optimized GPU code. This fuses multiple
# operations into single GPU kernels, reducing memory bandwidth usage and overhead.
# First call is slow (compilation), but subsequent calls are much faster.
model = torch.compile(model, dynamic=False)

# Create the data pipeline and prefetch the first batch
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # Prefetch first batch while we finish setup

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Learning rate schedules
# ---------------------------------------------------------------------------
# All schedules are based on "progress" = fraction of the time budget elapsed (0.0 to 1.0)

def get_lr_multiplier(progress):
    """Learning rate schedule: warmup -> constant -> cooldown.

    The learning rate follows a trapezoidal shape:
    1. Warmup (0 to WARMUP_RATIO): LR ramps from 0 to 1x (currently disabled, ratio=0)
    2. Constant (WARMUP_RATIO to 1-WARMDOWN_RATIO): LR stays at 1x
    3. Cooldown (last WARMDOWN_RATIO of training): LR decays linearly to FINAL_LR_FRAC

    With defaults (warmup=0.0, warmdown=0.5, final=0.0):
    - First 50% of training: full learning rate
    - Last 50%: linear decay from full LR down to zero

    The cooldown is important: it lets the model "settle" into a good minimum.
    Without it, the model keeps taking large steps and never converges cleanly.
    """
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    """Muon momentum schedule: ramp from 0.85 to 0.95 over the first 300 steps.

    Lower momentum early in training makes the optimizer more responsive to initial
    gradients (which are noisy because the model hasn't learned much yet). Higher
    momentum later provides more stable, averaged gradient estimates.
    """
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    """Weight decay schedule: linearly decay from WEIGHT_DECAY to 0 during training.

    Weight decay acts as regularization (prevents the model from memorizing training
    data). It's most useful early in training when the model is learning general
    patterns. Near the end, we want the model to fine-tune its weights without the
    regularization penalty pulling them toward zero.
    """
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
# This is the main loop where the model actually learns. It runs for exactly
# TIME_BUDGET seconds (5 minutes), doing as many optimization steps as possible.

t_start_training = time.time()
smooth_train_loss = 0      # Exponential moving average of training loss (for logging)
total_training_time = 0    # Accumulated wall-clock training time (excludes first 10 warmup steps)
step = 0

while True:
    torch.cuda.synchronize()  # Wait for all GPU work to finish before timing
    t0 = time.time()

    # --- Gradient accumulation loop ---
    # Process multiple micro-batches, accumulating gradients before one optimizer step.
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:         # Run in bfloat16 mixed precision
            loss = model(x, y)     # Forward pass: compute predictions and loss
        train_loss = loss.detach() # Save loss value for logging (detach from computation graph)
        loss = loss / grad_accum_steps  # Scale loss so gradients average correctly across micro-steps
        loss.backward()            # Backward pass: compute gradients of all parameters
        x, y, epoch = next(train_loader)  # Prefetch next batch while GPU computes

    # --- Update learning rates and other schedules based on training progress ---
    progress = min(total_training_time / TIME_BUDGET, 1.0)  # 0.0 to 1.0
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm  # Scale each group's LR by the schedule
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    # --- Optimizer step: update all model parameters using accumulated gradients ---
    optimizer.step()
    model.zero_grad(set_to_none=True)  # Clear gradients for next step (set_to_none saves memory)

    train_loss_f = train_loss.item()  # Convert from GPU tensor to Python float

    # Fast fail: if loss explodes (e.g., due to bad hyperparameters), abort immediately
    # rather than wasting the remaining 5 minutes
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()  # Wait for all GPU work to finish before timing
    t1 = time.time()
    dt = t1 - t0  # Wall-clock time for this step

    # Don't count the first 10 steps toward the time budget because they include
    # torch.compile JIT compilation overhead (which can be 30+ seconds).
    # This ensures the time budget measures actual training time, not setup time.
    if step > 10:
        total_training_time += dt

    # --- Logging ---
    # Exponential moving average (EMA) of training loss for smooth logging.
    # Raw loss fluctuates a lot step-to-step; EMA smooths it out.
    ema_beta = 0.9  # Higher = smoother but slower to reflect recent changes
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    # Debias: correct for the fact that EMA starts at 0 (underestimates early values)
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)  # Training throughput
    # MFU = Model FLOPs Utilization: what % of the GPU's theoretical peak we're achieving
    # Higher MFU = more efficient use of hardware. Good values: 30-50% on H100.
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    # Print progress on a single line (using \r to overwrite in-place)
    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # --- Garbage collection management ---
    # Python's garbage collector can cause ~500ms stalls during training (it pauses
    # everything to scan for unreachable objects). To avoid this:
    # - After step 0: do one final GC, freeze all surviving objects (so GC ignores
    #   them in future), then disable automatic GC entirely.
    # - Every 5000 steps: do a manual GC pass to clean up any accumulated garbage.
    if step == 0:
        gc.collect()
        gc.freeze()    # Mark all current objects as permanent (GC won't scan them)
        gc.disable()   # Disable automatic garbage collection
    elif (step + 1) % 5000 == 0:
        gc.collect()   # Periodic manual cleanup

    step += 1

    # Check if we've used up the 5-minute time budget.
    # Only check after step 10 (after compilation warmup is done).
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # Newline after the \r-overwritten training log line

total_tokens = step * TOTAL_BATCH_SIZE  # Total tokens processed during training

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------
# Switch model to evaluation mode (disables dropout, etc. — though this model
# doesn't use dropout, it's good practice) and compute the validation BPB metric.

model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# ---------------------------------------------------------------------------
# Final summary — the AI agent reads this output to decide if the experiment
# improved the model (lower val_bpb = better).
# ---------------------------------------------------------------------------

t_end = time.time()
startup_time = t_start_training - t_start  # Time spent on setup before training started
# Steady-state MFU: exclude the first 10 warmup steps for a cleaner efficiency estimate
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024  # Peak GPU memory usage

print("---")
print(f"val_bpb:          {val_bpb:.6f}")           # THE KEY METRIC — lower is better
print(f"training_seconds: {total_training_time:.1f}") # Should be ~300s (5 minutes)
print(f"total_seconds:    {t_end - t_start:.1f}")     # Including setup + eval overhead
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")        # GPU memory used
print(f"mfu_percent:      {steady_state_mfu:.2f}")    # Hardware utilization efficiency
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")  # Millions of tokens processed
print(f"num_steps:        {step}")                     # Number of optimizer steps completed
print(f"num_params_M:     {num_params / 1e6:.1f}")    # Model size in millions of parameters
print(f"depth:            {DEPTH}")                    # Number of transformer layers
