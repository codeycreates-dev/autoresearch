"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.

--- HIGH-LEVEL OVERVIEW ---

This script does two things before you can train a language model:

1. DOWNLOAD DATA: The training text is stored on HuggingFace as thousands of
   "shards" (small files). Each shard is a Parquet file (a columnar data format
   popular in data engineering) containing many text documents. This script
   downloads as many shards as you specify.

2. TRAIN A TOKENIZER: Language models don't read raw text — they work with
   numbers (called "tokens"). A tokenizer converts text -> numbers and back.
   This script trains a BPE (Byte Pair Encoding) tokenizer from scratch on the
   downloaded data.

This script also provides runtime utilities that train.py imports:
   - A Tokenizer wrapper class for encoding/decoding text
   - A dataloader that feeds batches of tokenized text to the model during training
   - An evaluation function that measures model quality in "bits per byte" (BPB)
"""

import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq  # Library for reading Parquet columnar data files
import rustbpe        # Rust-based BPE tokenizer trainer (fast implementation)
import tiktoken       # OpenAI's tokenizer library — used here as the runtime tokenizer format
import torch          # PyTorch — the deep learning framework

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

# Context length: how many tokens the model can "see" at once. Think of it like
# the model's short-term memory window. 2048 tokens ≈ roughly 1500 words.
MAX_SEQ_LEN = 2048

# Each training run gets exactly 5 minutes (300 seconds) of wall-clock training
# time. This fixed budget means experiments are comparable regardless of what
# the AI agent changes (model size, batch size, etc.).
TIME_BUDGET = 300

# How many tokens to use when evaluating model quality on the validation set.
# More tokens = more stable/reliable measurement, but takes longer.
# 40 * 524288 ≈ 21 million tokens.
EVAL_TOKENS = 40 * 524288

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All downloaded data and trained tokenizer files go into this cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# The training data lives on HuggingFace as a dataset called "climbmix-400b-shuffle".
# It's a large (~400 billion token) shuffled mix of text from various sources.
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"

# The dataset is split into 6543 shards (numbered 0 to 6542).
# Each shard is a Parquet file containing thousands of text documents.
MAX_SHARD = 6542

# The very last shard is reserved as the "validation" set — data the model
# never trains on, used only to measure how well it generalizes.
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

# The tokenizer's vocabulary size — how many unique "tokens" (word pieces) it knows.
# 8192 is relatively small (GPT-4 uses ~100K). Smaller vocab = each token represents
# less text, so you need more tokens per document, but the model's embedding table
# (a big lookup table mapping token IDs to vectors) is smaller.
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3}).
# This regex controls how raw text is pre-split before BPE merges are applied.
# For example, it keeps contractions like "don't" as one chunk, handles
# whitespace intelligently, and limits number sequences to 2 digits at a time
# (so "12345" becomes "12" "34" "5" rather than one token).
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Special tokens are tokens with special meaning that don't appear in normal text.
# Here we reserve 4 of them. The first one (reserved_0) is used as BOS
# (Beginning Of Sequence) — a marker placed at the start of each document so the
# model knows "a new document starts here."
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success.

    Each shard is ~4-5 MB. Downloads to a .tmp file first, then renames to the
    final name — this prevents half-downloaded corrupt files from being mistaken
    for complete ones. If a download fails, it retries up to 5 times with
    exponential backoff (wait 2s, 4s, 8s, 16s between retries).
    """
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    # Skip if already downloaded
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            # stream=True means don't load the whole file into memory at once;
            # instead, read it in 1MB chunks (important for large files)
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise exception if HTTP error (404, 500, etc.)
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Atomic rename: the file only appears with its real name once fully written
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial downloads
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            # Exponential backoff: wait longer between each retry (2s, 4s, 8s, 16s)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard.

    Uses Python's multiprocessing Pool to download multiple shards in parallel
    (default 8 at a time). Always ensures the validation shard is downloaded,
    even if you only request a small number of training shards.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    # Always include the validation shard (the last one) even if not in the range
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    # Count what's already downloaded to avoid re-downloading
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    # Download in parallel using a process pool
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory.
    Excludes any .tmp files (partially downloaded shards)."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard).

    This is a "generator" that lazily reads documents one-by-one from the Parquet
    files. Used to feed text into the BPE tokenizer trainer. Each document is
    capped at 10,000 characters to prevent extremely long documents from dominating
    the tokenizer's vocabulary. Stops after reading 1 billion characters total.

    Parquet files are organized into "row groups" — internal chunks that allow
    efficient partial reads without loading the entire file into memory.
    """
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train a BPE (Byte Pair Encoding) tokenizer using rustbpe, save as tiktoken pickle.

    HOW BPE WORKS (simplified):
    1. Start with individual characters (or bytes) as your initial vocabulary.
    2. Count every pair of adjacent tokens in the training text.
    3. Merge the most frequent pair into a single new token.
    4. Repeat steps 2-3 until you reach the desired vocabulary size.

    For example, if "th" appears very frequently, it becomes a single token.
    Then if "the" is frequent, it merges too. Common words like "the" end up
    as single tokens, while rare words get split into multiple tokens.

    The result is a vocabulary of 8192 "subword" tokens that balances between:
    - Character-level (tiny vocab, many tokens per word) — too slow
    - Word-level (huge vocab, one token per word) — can't handle new words

    After training, we also build a "token_bytes" lookup table that stores how
    many UTF-8 bytes each token represents. This is needed for the BPB (bits
    per byte) evaluation metric, which measures model quality in a way that's
    independent of vocabulary size.
    """
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    # Skip if already trained
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    # --- Train with rustbpe (a fast Rust-based BPE trainer) ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    # Reserve slots for special tokens by subtracting them from the total vocab size
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    # Train BPE on the text data using the GPT-4-style split pattern
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build a tiktoken Encoding object from the trained BPE merges.
    # tiktoken is OpenAI's fast tokenizer runtime — we use it because it's
    # efficient for encoding/decoding during training, even though we trained
    # with rustbpe.
    pattern = tokenizer.get_pattern()
    # "mergeable_ranks" is the core of a BPE tokenizer: a dictionary mapping
    # byte sequences to their rank (priority) for merging. Lower rank = merged first.
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    # Special tokens get IDs after all the regular tokens
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save the trained tokenizer as a pickle file for fast loading later
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    # For each token ID in the vocabulary, compute how many UTF-8 bytes it represents.
    # Example: token "the" = 3 bytes, token "é" = 2 bytes (in UTF-8), special tokens = 0 bytes.
    # This is needed because the evaluation metric (BPB) measures quality in bits per byte,
    # which is vocabulary-size-independent — so changing vocab size doesn't unfairly affect the score.
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)  # Special tokens don't represent real text bytes
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check: encode some text and decode it back — should be identical
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper for encoding text -> token IDs and decoding back.

    This class wraps a tiktoken Encoding object and provides a clean interface
    for the training code. The actual tokenizer was trained above; this class
    just loads and uses it.

    Key concept: Encoding converts text like "Hello world" into a list of
    integer IDs like [1234, 567]. The model works entirely with these numbers.
    Decoding converts the numbers back to text.
    """

    def __init__(self, enc):
        self.enc = enc
        # Pre-compute the BOS (Beginning Of Sequence) token ID for fast access
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        """Load a previously trained tokenizer from disk."""
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        """Returns total number of tokens in the vocabulary (regular + special)."""
        return self.enc.n_vocab

    def get_bos_token_id(self):
        """Returns the integer ID for the Beginning-Of-Sequence token."""
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        """Convert text to a list of token IDs.

        Args:
            text: A string or list of strings to encode.
            prepend: Optional token to prepend (e.g., BOS token). Can be an int
                     (token ID) or a string (token name).
            num_threads: Number of threads for batch encoding (when text is a list).

        Returns:
            A list of ints (if text is a string) or list of lists of ints
            (if text is a list of strings).
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            # encode_ordinary skips special token handling for speed
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            # Batch encoding: process multiple strings in parallel using threads
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        """Convert a list of token IDs back to text."""
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    """Load the precomputed tensor mapping each token ID -> its byte length.
    Used during evaluation to compute the bits-per-byte metric."""
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files.

    This generator yields batches of raw text documents from either the training
    or validation split, forever (looping back to the beginning when all documents
    are exhausted — each full pass through the data is called an "epoch").

    Args:
        split: "train" (all shards except validation) or "val" (only the validation shard).
        tokenizer_batch_size: How many documents to yield at once (for efficient
                              batch tokenization).

    Yields:
        Tuples of (batch_of_texts, epoch_number). The epoch number increments
        each time we loop through the entire dataset.
    """
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:  # Loop forever — the training loop decides when to stop
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                # Yield in chunks of tokenizer_batch_size for efficient encoding
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """BOS-aligned dataloader with best-fit packing.

    This is one of the most important functions in the codebase. It prepares
    batches of tokenized text for the model to train on.

    HOW LANGUAGE MODEL TRAINING WORKS:
    The model sees a sequence of tokens and tries to predict the next one.
    For input  [A, B, C, D], the targets are [B, C, D, E].
    So "inputs" is the sequence shifted left by one position relative to "targets".

    WHAT THIS DATALOADER DOES:
    1. Reads raw text documents from parquet files
    2. Tokenizes them (text -> integer IDs)
    3. Packs multiple documents into fixed-length rows using "best-fit" packing
    4. Returns batches of (inputs, targets) tensors for training

    BEST-FIT PACKING EXPLAINED:
    Each row has a fixed capacity of T+1 tokens. Documents vary in length.
    Instead of padding short documents with zeros (wasteful), we pack multiple
    documents into each row like a bin-packing problem:
    - Each document starts with a BOS token to mark "new document starts here"
    - We greedily pick the largest document that fits the remaining space
    - When nothing fits, we crop the shortest available document to fill exactly
    - This achieves 100% utilization — no wasted padding tokens

    Example (T=10, so row capacity = 11 tokens):
    Row: [BOS doc1_tok1 doc1_tok2 BOS doc2_tok1 doc2_tok2 doc2_tok3 doc2_tok4 BOS doc3_tok1 doc3_tok2]
    Then: inputs  = row[:-1]  (first 10 tokens)
          targets = row[1:]   (last 10 tokens, shifted by 1)

    Args:
        tokenizer: The Tokenizer object for encoding text.
        B: Batch size — how many rows (sequences) per batch.
        T: Sequence length — how many tokens per row the model sees.
        split: "train" or "val".
        buffer_size: How many tokenized documents to keep in memory for packing.

    Yields:
        Tuples of (inputs, targets, epoch):
        - inputs: [B, T] tensor of token IDs (what the model sees)
        - targets: [B, T] tensor of token IDs (what the model should predict)
        - epoch: current epoch number (how many times we've looped through data)
    """
    assert split in ["train", "val"]
    row_capacity = T + 1  # +1 because inputs and targets overlap by T tokens
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []  # Buffer of tokenized documents waiting to be packed into rows
    epoch = 1

    def refill_buffer():
        """Fetch the next batch of documents, tokenize them, add to buffer."""
        nonlocal epoch
        doc_batch, epoch = next(batches)
        # Encode all documents and prepend BOS token to each
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate GPU and CPU buffers for efficiency.
    # "pin_memory" makes CPU->GPU transfers faster by using page-locked memory.
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    # Split buffers into input and target views (they share the same underlying memory)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        # Pack B rows, each of length row_capacity
        for row_idx in range(B):
            pos = 0  # Current position in this row
            while pos < row_capacity:
                # Keep the buffer stocked with enough documents for efficient packing
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # BEST-FIT PACKING: find the largest document that fits entirely
                # in the remaining space. This minimizes wasted space.
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    # Found a document that fits — place it in the row
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No document fits the remaining space — crop the shortest
                    # document to fill exactly. We pick the shortest to minimize
                    # the amount of text we throw away.
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Split each row into input (first T tokens) and target (last T tokens)
        # Input:  [tok_0, tok_1, ..., tok_{T-1}]
        # Target: [tok_1, tok_2, ..., tok_T]
        # The model learns to predict each target token given all preceding input tokens.
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        # Transfer from CPU -> GPU asynchronously (non_blocking=True means the CPU
        # doesn't wait for the transfer to finish before continuing)
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()  # Disable gradient tracking — saves memory during evaluation
def evaluate_bpb(model, tokenizer, batch_size):
    """Compute Bits Per Byte (BPB) on the validation set.

    WHY BITS PER BYTE?
    The most common metric for language models is "perplexity" or "cross-entropy
    loss", but these depend on vocabulary size. If you change vocab from 8K to
    16K tokens, the loss numbers aren't directly comparable because each token
    represents a different amount of text.

    BPB solves this by measuring: "on average, how many bits does the model need
    to encode each byte of text?" This is independent of tokenization choices.

    HOW IT WORKS:
    1. For each token the model predicts, compute the cross-entropy loss (in nats).
       This measures how "surprised" the model is by the correct answer.
    2. For each target token, look up how many UTF-8 bytes it represents.
    3. Sum up all the losses and all the bytes across the validation set.
    4. Convert from nats to bits (multiply by 1/ln(2)) and divide by total bytes.

    Lower BPB = better model. For reference:
    - Random guessing on English text: ~8 BPB
    - Good language model: ~1.0 BPB
    - State of the art: ~0.7-0.8 BPB

    Special tokens (BOS markers) are excluded from the calculation since they
    don't represent real text content.
    """
    # Load the precomputed table mapping token_id -> number of UTF-8 bytes
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0  # Accumulated cross-entropy loss (in nats, i.e., natural log units)
    total_bytes = 0    # Accumulated byte count of all target tokens
    for _ in range(steps):
        x, y, _ = next(val_loader)  # x = inputs, y = targets
        # Get per-token loss (reduction='none' means don't average — return loss for each token)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        # Look up how many bytes each target token represents
        nbytes = token_bytes[y_flat]
        # Mask out special tokens (which have 0 bytes)
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    # Convert nats to bits: 1 bit = ln(2) nats, so bits = nats / ln(2)
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8,
                        help="Number of parallel download workers")
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data shards from HuggingFace
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train BPE tokenizer on the downloaded data
    train_tokenizer()
    print()
    print("Done! Ready to train.")
