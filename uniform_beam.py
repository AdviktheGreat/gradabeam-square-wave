"""Control: the paper's chunked beam search with a uniform-random proposal, on this oracle.

Evo 2's inference-time search (SI A.6.1) keeps K' partial designs; each step samples K 128-bp chunks per beam,
scores every candidate (loss masked to the designed region so far) and keeps the K' best. The paper's own
"uniform generator" curve uses a uniform-over-ACGT proposal in place of Evo 2. Running that here, on the same
oracle, checks that this pipeline reproduces the paper's search behaviour and gives a non-learned reference
for GradaBeam at an equal number of scored candidates (K' * K * 153 chunks).

    PYTHONPATH=. python uniform_beam.py --kprime 2 --k 15 --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from squarewave.remote import BASES, RemoteOracle

CHUNK = 128
DESIGN_BP = 19_584
OUT = Path(__file__).parent / "results" / "uniform_beam"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kprime", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(5000 + a.seed)
    ro = RemoteOracle()
    beams = [""] * a.kprime
    n_scored, t0 = 0, time.time()
    n_chunks = DESIGN_BP // CHUNK
    for step in range(n_chunks):
        cands = []
        for b in beams:
            for _ in range(a.k):
                cands.append(b + "".join(rng.choice(list(BASES), size=CHUNK)))
        loss = ro.score(cands)["loss"]
        n_scored += len(cands)
        keep = np.argsort(loss, kind="stable")[: a.kprime]
        beams = [cands[i] for i in keep]
        if step % 10 == 0 or step == n_chunks - 1:
            print(f"[ubeam K'={a.kprime} K={a.k} s{a.seed}] chunk {step + 1}/{n_chunks} scored {n_scored} best partial loss {loss[keep[0]]:.2f} "
                  f"wall {time.time() - t0:.0f}s ~${ro.usd_estimate:.2f}", flush=True)
    final = ro.score(beams[:1])["loss"][0]
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"ubeam_kp{a.kprime}_k{a.k}_s{a.seed}"
    (OUT / f"{name}.json").write_text(
        json.dumps({"kprime": a.kprime, "k": a.k, "seed": a.seed, "tok_per_bp": a.kprime * a.k, "n_scored": n_scored,
                    "final_loss": float(final), "seq": beams[0], "gpu_s": ro.gpu_s, "wall_s": time.time() - t0})
    )
    print(f"[{name}] done: final loss {final:.2f} scored {n_scored}", flush=True)


if __name__ == "__main__":
    main()
