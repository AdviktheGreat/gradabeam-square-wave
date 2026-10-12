"""Gradient robustness: loss-scale backoff and agreement between loss scales."""
from pathlib import Path
import numpy as np
from squarewave.remote import BASES, RemoteOracle

ro = RemoteOracle()
seqs = {p.parent.name: p.read_text() for p in sorted(Path("results/runs").glob("pilot_*/best.txt"))}
rng = np.random.default_rng(5)
seqs["random"] = "".join(rng.choice(list(BASES), size=19584))
for name, s in seqs.items():
    r = ro.score_and_grad([s])
    outs = {}
    for sc in (1.0, 64.0, 4096.0):
        o = ro.bor.score_and_grad.remote([s], open_bp=1664, closed_bp=1664, loss_scale=sc)
        outs[sc] = (np.frombuffer(o["grad"], dtype=np.float32), o["scale"], o["finite"])
    ref = outs[64.0][0]
    cos = {sc: float(np.dot(g, ref) / (np.linalg.norm(g) * np.linalg.norm(ref) + 1e-30)) if np.isfinite(g).all() else None for sc, (g, _, _) in outs.items()}
    print(f"{name:18s} loss {r['loss'][0]:.2f} default-scale-used {r['loss_scale']:.0f} | used {[o[1] for o in outs.values()]} | cos vs scale64 {cos}")
