import time
import numpy as np
from squarewave.remote import BASES, RemoteOracle
n = 19_584
rng = np.random.default_rng(2)
mk = lambda: "".join(rng.choice(list(BASES), size=n))
ro = RemoteOracle()
ro.score([mk()]); ro.score_and_grad([mk()])  # warm
for kind in ("score", "grad"):
    walls, gpus = [], []
    for _ in range(12):
        d = [mk()]
        g0 = sum(ro.gpu_s.values()); t = time.time()
        (ro.score if kind == "score" else ro.score_and_grad)(d)
        walls.append(time.time() - t); gpus.append(sum(ro.gpu_s.values()) - g0)
    print(f"{kind}: wall median {np.median(walls):.2f}s (min {min(walls):.2f}, max {max(walls):.2f}); enf+bor gpu median {np.median(gpus):.2f}s")
