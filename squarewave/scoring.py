"""Scoring math for the Evo 2 (Brixi et al. 2026, SI A.6) chromatin-accessibility design task.

Pure numpy, so the same functions validate published sequences and score GrAdaBeam designs.
Geometry: the designed region starts exactly at the left edge of each predictor's output window.
"""

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

DATA = Path(__file__).resolve().parent.parent / "data"

ENF_CTX, ENF_UP, ENF_BIN, ENF_BINS = 196_608, 40_960, 128, 896
BOR_CTX, BOR_UP, BOR_BIN, BOR_BINS = 524_288, 163_840, 32, 6_144

# 0-based mouse track indices for "DNASE:129 ES-E14" (matched by description, see targets lists)
ENF_MOUSE_TRACK = 11
BOR_MOUSE_TRACK = 741

LONG_WAVE_OPEN_BP = 1_664
LONG_WAVE_CLOSED_BP = 1_664


def load_flanks() -> tuple[str, str]:
    up = (DATA / "mm39_chrX_flank_up163840.txt").read_text().strip()
    down = (DATA / "mm39_chrX_flank_down400000.txt").read_text().strip()
    assert len(up) == BOR_UP and len(down) == 400_000
    return up, down


def load_published_long_wave() -> str:
    """Evo 2 long square wave from Supplementary Data 3 (CC BY-NC-ND 4.0), not redistributed with this repo."""
    path = DATA / "published_dnase_squarewave_long_20kb.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. See data/README.md for how to obtain the Evo 2 published sequence.")
    return path.read_text().strip()


def assemble(design: str, up: str, down: str, ctx: int, n_up: int) -> str:
    """Native upstream flank + design + native downstream fill (downstream begins after the replaced region)."""
    n_down = ctx - n_up - len(design)
    assert n_down > 0 and len(up) >= n_up and len(down) >= n_down
    out = up[len(up) - n_up :] + design + down[:n_down]
    assert len(out) == ctx
    return out


def wave_pattern_bp(length_bp: int, open_bp: int, closed_bp: int) -> np.ndarray:
    """Per-bp binary target, open chromatin first."""
    pos = np.arange(length_bp) % (open_bp + closed_bp)
    return (pos < open_bp).astype(np.float64)


def pattern_on_bins(pattern_bp: np.ndarray, bin_bp: int) -> np.ndarray:
    assert len(pattern_bp) % bin_bp == 0
    return pattern_bp.reshape(-1, bin_bp).mean(axis=1)


def enformer_profile(pred: np.ndarray) -> np.ndarray:
    """pred: (896,) raw track. Divide by the max over the whole output window (SI A.6.2)."""
    return pred / pred.max()


def borzoi_profile(pred: np.ndarray) -> np.ndarray:
    """pred: (4, 6144) raw replicates -> lower confidence bound mean - std after global max-normalization."""
    p = pred / pred.max()
    return p.mean(axis=0) - p.std(axis=0)


def l1_loss(profile: np.ndarray, target_bins: np.ndarray) -> float:
    n = len(target_bins)
    return float(np.abs(target_bins - profile[:n]).sum())


def design_loss(enf_pred: np.ndarray, bor_pred: np.ndarray, target_bp: np.ndarray) -> dict:
    t_e = pattern_on_bins(target_bp, ENF_BIN)
    t_b = pattern_on_bins(target_bp, BOR_BIN)
    pe, pb = enformer_profile(enf_pred), borzoi_profile(bor_pred)
    fe, fb = l1_loss(pe, t_e), l1_loss(pb, t_b)
    return {"loss": 0.5 * (fe + fb), "f_enformer": fe, "f_borzoi": fb, "profile_enformer": pe, "profile_borzoi": pb}


def auroc(profile: np.ndarray, target_bins: np.ndarray) -> float:
    y = target_bins > 0.5
    s = profile[: len(y)]
    ranks = rankdata(s)
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def design_auroc(profile_enf: np.ndarray, profile_bor: np.ndarray, target_bp: np.ndarray) -> dict:
    a_e = auroc(profile_enf, pattern_on_bins(target_bp, ENF_BIN))
    a_b = auroc(profile_bor, pattern_on_bins(target_bp, BOR_BIN))
    return {"auroc": 0.5 * (a_e + a_b), "auroc_enformer": a_e, "auroc_borzoi": a_b}


def icc2k(ratings: np.ndarray) -> float:
    """ICC(2,k) absolute agreement; ratings shape (n_targets, k_raters)."""
    n, k = ratings.shape
    grand = ratings.mean()
    ss_rows = k * ((ratings.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((ratings.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((ratings - grand) ** 2).sum() - ss_rows - ss_cols
    ms_r, ms_c, ms_e = ss_rows / (n - 1), ss_cols / (k - 1), ss_err / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n))


def ensemble_icc(enf_pred: np.ndarray, bor_pred: np.ndarray, n_design_bp: int) -> float:
    """Agreement between Enformer (upsampled 4x to 32 bp) and the 4 Borzoi replicates over the design."""
    n32 = n_design_bp // BOR_BIN
    e = np.repeat(enf_pred / enf_pred.max(), ENF_BIN // BOR_BIN)[:n32]
    b = (bor_pred / bor_pred.max())[:, :n32]
    return icc2k(np.vstack([e[None, :], b]).T)
