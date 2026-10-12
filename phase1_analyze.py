"""Phase 1b: score the cached Enformer/Borzoi outputs with the paper's loss, AUROC and ICC."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from squarewave import scoring as S  # noqa: E402

d = np.load(Path(__file__).parent / "results" / "phase1_raw.npz")
names, enf, bor = list(d["names"]), d["enf"], d["bor"]
enf_tracks, bor_tracks = list(d["enf_tracks"]), list(d["bor_tracks"])
n_design = 19_584
target = S.wave_pattern_bp(n_design, S.LONG_WAVE_OPEN_BP, S.LONG_WAVE_CLOSED_BP)
t_e, t_b = S.pattern_on_bins(target, S.ENF_BIN), S.pattern_on_bins(target, S.BOR_BIN)
print("timing (enformer s, borzoi s):", d["timing"])
print(f"target bins: enformer {len(t_e)} (open frac {t_e.mean():.3f}), borzoi {len(t_b)}")

print("\nPer-track AUROC / L1 loss (design region only)")
print(f"{'design':24s} " + " ".join(f"E{t:<5d}" for t in enf_tracks) + " | " + " ".join(f"B{t:<5d}" for t in bor_tracks))
for i, name in enumerate(names):
    ea, eb = [], []
    for j in range(len(enf_tracks)):
        p = S.enformer_profile(enf[i][:, j])
        ea.append(f"{S.auroc(p, t_e):.3f}")
    for j in range(len(bor_tracks)):
        p = S.borzoi_profile(bor[i][:, j, :])
        eb.append(f"{S.auroc(p, t_b):.3f}")
    print(f"{name:24s} " + " ".join(f"{x:6s}" for x in ea) + " | " + " ".join(f"{x:6s}" for x in eb))

print("\nL1 losses")
for i, name in enumerate(names):
    le = [f"{S.l1_loss(S.enformer_profile(enf[i][:, j]), t_e):6.1f}" for j in range(len(enf_tracks))]
    lb = [f"{S.l1_loss(S.borzoi_profile(bor[i][:, j, :]), t_b):6.1f}" for j in range(len(bor_tracks))]
    print(f"{name:24s} " + " ".join(le) + " | " + " ".join(lb))

je, jb = enf_tracks.index(S.ENF_MOUSE_TRACK), bor_tracks.index(S.BOR_MOUSE_TRACK)
print(f"\nCombined score with Enformer mouse track {S.ENF_MOUSE_TRACK} and Borzoi mouse track {S.BOR_MOUSE_TRACK}")
for i, name in enumerate(names):
    e, b = enf[i][:, je], bor[i][:, jb, :]
    loss = S.design_loss(e, b, target)
    au = S.design_auroc(loss["profile_enformer"], loss["profile_borzoi"], target)
    icc = S.ensemble_icc(e, b, n_design)
    print(
        f"{name:24s} loss={loss['loss']:.2f} (enf {loss['f_enformer']:.2f}, bor {loss['f_borzoi']:.2f}) "
        f"AUROC={au['auroc']:.3f} (enf {au['auroc_enformer']:.3f}, bor {au['auroc_borzoi']:.3f}) ICC={icc:.3f}"
    )
