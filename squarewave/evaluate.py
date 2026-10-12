"""Score designs with the paper's metrics, plus held-out-track and sequence-statistics checks."""

import numpy as np

from squarewave import scoring as S
from squarewave.metrics import design_stats, dinuc_freq
from squarewave.remote import RemoteOracle

# Same-cell-type accessibility tracks the optimizer never sees (mouse head indices, matched by description).
HELDOUT_ENF = [10, 44, 45, 46]  # DNASE: 129 ES-CJ7; ZHBTc4-mESC (untreated, +dox 24 h, +dox 6 h)
HELDOUT_BOR = [740, 774, 775, 776]
BATCH = 8


def reference_dinuc() -> np.ndarray:
    return dinuc_freq((S.DATA / "mm39_chrX_native_region.txt").read_text().strip())


def _chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def evaluate(ro: RemoteOracle, seqs: list[str]) -> list[dict]:
    n = len(seqs[0])
    target = S.wave_pattern_bp(n, ro.open_bp, ro.closed_bp)
    ref = reference_dinuc()
    t_e, t_b = S.pattern_on_bins(target, S.ENF_BIN), S.pattern_on_bins(target, S.BOR_BIN)
    rows = []
    for batch in _chunks(seqs, BATCH):
        main = ro.score(batch, return_raw=True)
        held = ro.raw_tracks(batch, HELDOUT_ENF, HELDOUT_BOR)
        for i, seq in enumerate(batch):
            d = S.design_loss(main["raw_enformer"][i], main["raw_borzoi"][i], target)
            a = S.design_auroc(d["profile_enformer"], d["profile_borzoi"], target)
            ho_e = [S.auroc(S.enformer_profile(held["enformer"][i, k]), t_e) for k in range(len(HELDOUT_ENF))]
            ho_b = [S.auroc(S.borzoi_profile(held["borzoi"][i, :, k, :]), t_b) for k in range(len(HELDOUT_BOR))]
            rows.append(
                {
                    "loss": d["loss"],
                    "f_enformer": d["f_enformer"],
                    "f_borzoi": d["f_borzoi"],
                    **a,
                    "icc": S.ensemble_icc(main["raw_enformer"][i], main["raw_borzoi"][i], n),
                    "heldout_auroc_enformer": float(np.mean(ho_e)),
                    "heldout_auroc_borzoi": float(np.mean(ho_b)),
                    "heldout_auroc": float(0.5 * (np.mean(ho_e) + np.mean(ho_b))),
                    **design_stats(seq, target, ref),
                }
            )
    return rows
