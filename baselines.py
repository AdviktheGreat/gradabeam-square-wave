"""Evaluate reference designs (Evo 2 published designs, random, native) with the full metric suite."""
import json
from pathlib import Path

import numpy as np

from squarewave import scoring as S
from squarewave.evaluate import evaluate
from squarewave.remote import BASES, RemoteOracle

D = Path("data")
pub = S.load_published_long_wave()
n = len(pub)
native = (D / "mm39_chrX_native_region.txt").read_text().strip()[:n]
rng = np.random.default_rng(0)
seqs = {
    "evo2_published_long_wave": pub,
    "uniform_random": "".join(rng.choice(list(BASES), size=n)),
    "native_mm39_region": native,
}
for p in sorted(Path("results/runs").glob("pilot_rate200/best.txt")):
    seqs["gradabeam_pilot_rate200_2k_fe"] = p.read_text()
ro = RemoteOracle()
rows = evaluate(ro, list(seqs.values()))
out = dict(zip(seqs, rows))
Path("results/baselines.json").write_text(json.dumps(out, indent=1))
keys = ["loss", "auroc", "auroc_enformer", "auroc_borzoi", "icc", "heldout_auroc", "gc_open", "gc_closed", "gc_wave_corr", "cpg_obs_exp", "dinuc_l1_vs_mm39", "zlib_ratio"]
print("%-32s" % "design" + " ".join("%10s" % k[:10] for k in keys))
for name, r in out.items():
    print("%-32s" % name + " ".join("%10.3f" % r[k] for k in keys))
