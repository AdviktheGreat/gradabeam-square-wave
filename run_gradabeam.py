"""Run GradaBeam on the long square wave task from a uniform-random start, under a forward-equivalent budget.

    PYTHONHASHSEED=<seed> PYTHONPATH=.:$GRADABEAM_PATH \
        python run_gradabeam.py --seed 0 --budget-fe 13000 --name main_s0

PYTHONHASHSEED matters: GradaBeam orders ties through Python set iteration, so results depend on the hash seed.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from gradabeam.gradabeam_optimizer import GradaBeam
from squarewave.budget import BudgetExhausted
from squarewave.gb_oracle import SquareWaveOracle
from squarewave.remote import RemoteOracle
from squarewave.starts import DESIGN_BP, make_start

RUNS = Path(__file__).parent / "results" / "runs"


def save(out: Path, oracle: SquareWaveOracle, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trace.npz", trace=np.array(oracle.trace))
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
    ap.add_argument("--beam-size", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--mut-rate", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-rollout-len", type=int, default=200)
    ap.add_argument("--no-pbt", action="store_true")
    ap.add_argument("--milestone-fe", type=float, default=250.0)
    ap.add_argument("--start", choices=["uniform", "markov"], default="uniform")
    a = ap.parse_args()

    meta = {**vars(a), "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "design_bp": DESIGN_BP}
    out = RUNS / a.name
    start = make_start(a.start, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "start.txt").write_text(start)

    remote = RemoteOracle()
    oracle = SquareWaveOracle(remote, budget_fe=a.budget_fe, milestone_fe=a.milestone_fe)
    last_save = time.time()
    steps = 0
    try:
        opt = GradaBeam(
            model_fn=oracle,
            start_sequence=start,
            mutations_per_sequence=a.mut_rate,
            beam_size=a.beam_size,
            n_rollouts_per_root=a.rollouts,
            exploration_alpha=a.alpha,
            use_pbt=not a.no_pbt,
            max_rollout_len=a.max_rollout_len,
            rng_seed=a.seed,
            eval_batch_size=1,
        )
        while True:
            opt.run(n_steps=1)
            steps += 1
            if time.time() - last_save > 60:
                save(out, oracle, {**meta, "steps": steps})
                print(
                    f"[{a.name}] steps {steps} fe {oracle.fe:.0f}/{a.budget_fe:.0f} (fwd {oracle.n_fwd}, grad {oracle.n_grad}) "
                    f"best loss {oracle.best_loss:.2f} wall {time.time() - oracle.t0:.0f}s ~${remote.usd_estimate:.2f}",
                    flush=True,
                )
                last_save = time.time()
    except BudgetExhausted:
        meta["finished"] = "budget"
    save(out, oracle, {**meta, "steps": steps})
    print(f"[{a.name}] done: fe {oracle.fe:.0f} best loss {oracle.best_loss:.2f} wall {time.time() - oracle.t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
