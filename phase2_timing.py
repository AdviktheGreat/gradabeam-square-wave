"""Warm latency and cost per oracle call on the custom Modal app."""
import time
from pathlib import Path

import numpy as np

from squarewave.remote import BASES, RemoteOracle

n = 19_584
rng = np.random.default_rng(1)
mk = lambda: "".join(rng.choice(list(BASES), size=n))
ro = RemoteOracle()
for bs in (1, 1, 1, 4, 8, 16):
    ds = [mk() for _ in range(bs)]
    g0 = dict(ro.gpu_s)
    t = time.time()
    ro.score(ds)
    dt = time.time() - t
    dg = {k: ro.gpu_s[k] - g0[k] for k in g0}
    print(f"score batch {bs:2d}: wall {dt:.2f}s ({dt/bs:.3f}/seq)  gpu enf {dg['enformer']:.2f} bor {dg['borzoi']:.2f}")
for _ in range(3):
    d = [mk()]
    g0 = dict(ro.gpu_s)
    t = time.time()
    ro.score_and_grad(d)
    dt = time.time() - t
    dg = {k: ro.gpu_s[k] - g0[k] for k in g0}
    print(f"grad  batch  1: wall {dt:.2f}s  gpu enf {dg['enformer']:.2f} bor {dg['borzoi']:.2f}")
print(f"est cost ${ro.usd_estimate:.3f}")
