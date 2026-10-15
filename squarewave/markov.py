"""Order-k Markov chain over ACGT trained on genome FASTA files, used to draw a genome-like GrAdaBeam start sequence.

Context of k bases -> distribution of the next base. Counts are collected on the forward strand only, upper-casing soft-masked
repeats, and k-mers that touch an N (or any non-ACGT) are skipped.
"""

import gzip
from pathlib import Path

import numpy as np

BASES = "ACGT"
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i
BLOCK = 16_000_000


def read_fasta_codes(path: Path) -> np.ndarray:
    """Single-record (possibly gzipped) FASTA -> uint8 array, A,C,G,T = 0..3 and 4 for anything else."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()
    header_end = raw.index(b"\n")
    assert raw.startswith(b">")
    return _LUT[np.frombuffer(raw[header_end + 1 :].translate(None, b"\n\r"), dtype=np.uint8)]


def count_kmers(codes: np.ndarray, order: int) -> np.ndarray:
    """(4**order, 4) counts of (context, next base), processed in blocks to bound memory."""
    counts = np.zeros(4**order * 4, dtype=np.int64)
    n = len(codes)
    for start in range(0, n - order, BLOCK):
        block = codes[start : min(start + BLOCK + order, n)]
        m = len(block) - order
        if m <= 0:
            continue
        bad = np.zeros(m, dtype=bool)
        ctx = np.zeros(m, dtype=np.int32)
        for j in range(order):
            part = block[j : j + m]
            bad |= part > 3
            ctx = ctx * 4 + part
        nxt = block[order : order + m]
        bad |= nxt > 3
        counts += np.bincount((ctx * 4 + nxt)[~bad], minlength=counts.size)
    return counts.reshape(4**order, 4)


def counts_to_probs(counts: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    p = counts.astype(np.float64) + pseudocount
    return p / p.sum(axis=1, keepdims=True)


def context_index(context: str) -> int:
    idx = 0
    for ch in context:
        idx = idx * 4 + BASES.index(ch)
    return idx


def sample(probs: np.ndarray, length: int, context: str, rng: np.random.Generator) -> str:
    """Autoregressively draw `length` bases given the preceding `order` bases in `context`."""
    order = int(round(np.log(probs.shape[0]) / np.log(4)))
    assert len(context) == order and set(context) <= set(BASES)
    cum = np.cumsum(probs, axis=1)
    mask = 4**order - 1
    idx = context_index(context)
    u = rng.random(length)
    out = np.empty(length, dtype=np.uint8)
    for i in range(length):
        b = int(np.searchsorted(cum[idx], u[i], side="right"))
        b = min(b, 3)
        out[i] = b
        idx = ((idx << 2) | b) & mask
    return "".join(BASES[b] for b in out)


def load_probs(path: Path) -> np.ndarray:
    return counts_to_probs(np.load(path)["counts"])
