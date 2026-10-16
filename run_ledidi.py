"""Run Ledidi on the long square wave task from a start sequence, under a forward-equivalent budget.

    PYTHONPATH=. python run_ledidi.py --seed 0 --budget-fe 13000 --l 0.01 --name ledidi_s0

Same oracle, start sequences, budget accounting and output files as run_gradabeam.py, so analyze_runs.py works on both.
The template Ledidi edits is the start sequence (a Markov-o5 sample by default). Ledidi's own early stopping is off so a
run always spends its whole budget, and max_iter is set from the budget.
"""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from ledidi import ledidi

from squarewave.budget import GRAD_FE, BudgetExhausted
from squarewave.ledidi_oracle import LedidiOracle, MeanLoss, to_onehot, to_strings
from squarewave.remote import RemoteOracle
from squarewave.starts import DESIGN_BP, make_start

RUNS = Path(__file__).parent / "results" / "runs"


def save(out: Path, oracle: LedidiOracle, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trace.npz", trace=np.array(oracle.trace), iters=np.array(oracle.iters))
    (out / "milestones.json").write_text(
        json.dumps({str(k): {"loss": v[0], "seq": v[1]} for k, v in oracle.milestones.items()})
    )
    (out / "best.txt").write_text(oracle.best_seq)
    meta = {
        **meta,
        "fe": oracle.fe,
        "n_fwd": oracle.n_fwd,
        "n_grad": oracle.n_grad,
        "n_bad_grad": oracle.n_bad_grad,
        "grad_loss_scales": {str(k): v for k, v in oracle.grad_scales.items()},
        "best_loss": oracle.best_loss,
        "wall_s": time.time() - oracle.t0,
        "gpu_s": oracle.remote.gpu_s,
        "usd_estimate": oracle.remote.usd_estimate,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget-fe", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--l", type=float, required=True, help="weight of Ledidi's edit-count (input) loss; library default 0.1")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=4, help="designs scored in parallel within one Ledidi step")
    ap.add_argument("--milestone-fe", type=float, default=250.0)
    ap.add_argument("--start", choices=["uniform", "markov"], default="markov")
    a = ap.parse_args()

    # The first model call scores the unedited template (1 fe); every later step scores batch_size designs with gradients.
    max_iter = int((a.budget_fe - 1) // (a.batch_size * GRAD_FE))
    meta = {**vars(a), "design_bp": DESIGN_BP, "max_iter": max_iter, "ledidi_version": version("ledidi"), "torch": torch.__version__}
    out = RUNS / a.name
    start = make_start(a.start, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "start.txt").write_text(start)

    remote = RemoteOracle(workers=2 * a.concurrency)

    def progress(o: LedidiOracle) -> None:
        save(out, o, {**meta, "steps": len(o.iters)})
        it = o.iters[-1] if o.iters else (0, float("nan"), float("nan"), float("nan"))
        print(
            f"[{a.name}] step {len(o.iters)}/{max_iter} fe {o.fe:.0f}/{a.budget_fe:.0f} best loss {o.best_loss:.2f} "
            f"batch mean {it[1]:.1f} min {it[2]:.1f} edits {it[3]:.0f} wall {time.time() - o.t0:.0f}s ~${remote.usd_estimate:.2f}",
            flush=True,
        )

    oracle = LedidiOracle(remote, start, a.budget_fe, a.concurrency, a.milestone_fe, on_progress=progress)
    try:
        batch = ledidi(
            oracle,
            to_onehot(start),
            torch.zeros(1, 1),
            device="cpu",
            random_state=a.seed,
            output_loss=MeanLoss(),
            l=a.l,
            batch_size=a.batch_size,
            lr=a.lr,
            max_iter=max_iter,
            early_stopping_iter=max_iter,
            verbose=False,
        )
        # Ledidi's own answer: the batch from the iteration with the lowest batch-mean total loss.
        (out / "ledidi_batch.txt").write_text("\n".join(to_strings(batch)))
        meta["finished"] = "budget"
    except BudgetExhausted:
        meta["finished"] = "budget_guard"
    save(out, oracle, {**meta, "steps": len(oracle.iters)})
    print(f"[{a.name}] done: fe {oracle.fe:.0f} best loss {oracle.best_loss:.2f} wall {time.time() - oracle.t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
