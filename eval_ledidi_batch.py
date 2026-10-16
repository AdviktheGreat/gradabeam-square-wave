"""Score the batch each Ledidi run returns (its own answer), as opposed to the best design seen during the run.

    MODAL_ENVIRONMENT=<env> PYTHONPATH=. python eval_ledidi_batch.py ledidi_markov_s0 ledidi_markov_s1 ...

Ledidi returns the batch of `batch_size` samples from the iteration with the lowest batch-mean total loss (paper loss plus
l times the edit count). The headline numbers instead take the best design out of every scored sample, which is how the GrAdaBeam
runs are scored. This checks that the headline does not depend on that selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from squarewave.evaluate import evaluate
from squarewave.remote import RemoteOracle

RUNS = Path(__file__).parent / "results" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/ledidi_batch_eval.json")
    a = ap.parse_args()

    ro = RemoteOracle()
    out = {}
    for run in a.runs:
        seqs = (RUNS / run / "ledidi_batch.txt").read_text().split()
        rows = evaluate(ro, seqs)
        pick = min(rows, key=lambda r: r["loss"])
        out[run] = {
            "n": len(rows),
            "batch_mean_loss": float(np.mean([r["loss"] for r in rows])),
            "batch_min_loss": pick["loss"],
            "batch_mean_auroc": float(np.mean([r["auroc"] for r in rows])),
            "batch_min_auroc": float(np.min([r["auroc"] for r in rows])),
            "batch_mean_heldout_auroc": float(np.mean([r["heldout_auroc"] for r in rows])),
            "best_in_batch": {k: pick[k] for k in ("loss", "auroc", "heldout_auroc", "icc", "cpg_obs_exp", "dinuc_l1_vs_mm39")},
        }
        print(run, {k: round(v, 3) for k, v in out[run].items() if isinstance(v, float)})
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
