"""Evaluate the best-so-far design of each run at fixed forward-equivalent budgets.

    PYTHONPATH=. python analyze_runs.py main_s0 main_s1 ... --fe 500 1000 2000 4000 8000 13000
"""

import argparse
import json
from pathlib import Path

import numpy as np

from squarewave.evaluate import evaluate
from squarewave.remote import RemoteOracle

RUNS = Path(__file__).parent / "results" / "runs"
KEYS = ["loss", "auroc", "auroc_enformer", "auroc_borzoi", "icc", "heldout_auroc", "gc_wave_corr", "dinuc_l1_vs_mm39"]


def best_at(run: str, fe: int) -> tuple[str, float, float]:
    """Best design the run had found by `fe` forward-equivalents (from the oracle's own trace), and its loss."""
    meta = json.loads((RUNS / run / "meta.json").read_text())
    if meta.get("finished") == "budget" and fe >= meta["budget_fe"]:
        return (RUNS / run / "best.txt").read_text(), meta["best_loss"], int(meta["fe"])
    marks = json.loads((RUNS / run / "milestones.json").read_text())
    ms = sorted(int(k) for k in marks)
    use = max((m for m in ms if m <= fe), default=None)
    if use is None:
        raise ValueError(f"{run}: no milestone at or below {fe}")
    seq = marks[str(use)]["seq"]
    return seq, marks[str(use)]["loss"], use


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fe", nargs="+", type=int, default=[500, 1000, 2000, 4000, 8000, 13000])
    ap.add_argument("--out", default="results/analysis.json")
    a = ap.parse_args()

    ro = RemoteOracle()
    jobs = []
    for run in a.runs:
        for fe in a.fe:
            try:
                seq, loss, used = best_at(run, fe)
            except ValueError:
                continue
            jobs.append((run, fe, used, loss, seq))
    rows = evaluate(ro, [j[4] for j in jobs])
    out = []
    print("%-14s %6s %6s " % ("run", "fe", "milest") + " ".join("%10s" % k[:10] for k in KEYS))
    for (run, fe, used, loss, _), r in zip(jobs, rows):
        # The re-score (plain scoring path) must agree with the optimizer's value (gradient path). Borzoi runs in fp16 autocast,
        # so the two differ in the last digits; the tolerance is relative because losses range from ~30 to ~200.
        assert abs(r["loss"] - loss) < max(0.05, 0.0025 * loss), (run, fe, r["loss"], loss)
        out.append({"run": run, "fe": fe, "milestone": used, "optimizer_loss": loss, **r})
        print("%-14s %6d %6d " % (run, fe, used) + " ".join("%10.3f" % r[k] for k in KEYS))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
