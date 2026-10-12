"""Sequence-level statistics used to judge whether a design is plausible or adversarial to the predictors."""

import itertools
import zlib

import numpy as np

DINUCS = ["".join(p) for p in itertools.product("ACGT", repeat=2)]


def dinuc_freq(seq: str) -> np.ndarray:
    counts = {d: 0 for d in DINUCS}
    for i in range(len(seq) - 1):
        d = seq[i : i + 2]
        if d in counts:
            counts[d] += 1
    v = np.array([counts[d] for d in DINUCS], dtype=float)
    return v / v.sum()


def cpg_obs_exp(seq: str) -> float:
    n = len(seq)
    c, g = seq.count("C"), seq.count("G")
    cg = sum(1 for i in range(n - 1) if seq[i] == "C" and seq[i + 1] == "G")
    return cg * n / max(c * g, 1)


def longest_homopolymer(seq: str) -> int:
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def window_gc(seq: str, win: int) -> np.ndarray:
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    gc = ((arr == ord("G")) | (arr == ord("C"))).astype(float)
    n = len(gc) // win
    return gc[: n * win].reshape(n, win).mean(1)


def design_stats(seq: str, target_bp: np.ndarray, reference_dinuc: np.ndarray) -> dict:
    """Naturalness-flavoured summary of one design against the open/closed wave target."""
    gc = window_gc(seq, 128)
    tgt = target_bp[: len(gc) * 128].reshape(-1, 128).mean(1)
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    is_gc = (arr == ord("G")) | (arr == ord("C"))
    open_mask = target_bp > 0.5
    cpg_windows = np.array(
        [sum(1 for j in range(i, i + 127) if seq[j] == "C" and seq[j + 1] == "G") for i in range(0, len(gc) * 128, 128)],
        dtype=float,
    )
    df = dinuc_freq(seq)
    return {
        "gc": float(is_gc.mean()),
        "gc_open": float(is_gc[open_mask].mean()),
        "gc_closed": float(is_gc[~open_mask].mean()),
        "gc_wave_corr": float(np.corrcoef(gc, tgt)[0, 1]),
        "cpg_per_128bp_wave_corr": float(np.corrcoef(cpg_windows, tgt)[0, 1]) if cpg_windows.std() > 0 else float("nan"),
        "cpg_obs_exp": cpg_obs_exp(seq),
        "dinuc_l1_vs_mm39": float(np.abs(df - reference_dinuc).sum()),
        "longest_homopolymer": longest_homopolymer(seq),
        "zlib_ratio": len(zlib.compress(seq.encode(), 9)) / len(seq),
    }
