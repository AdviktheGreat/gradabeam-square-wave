"""Before any GrAdaBeam run: how do uniform and Markov-o5 starts look to the oracle, and is the gradient any more
predictive of real single-base substitutions on genome-like DNA?

    MODAL_ENVIRONMENT=<env> PYTHONPATH=. python markov_start_check.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from run_gradabeam import DESIGN_BP, make_start
from squarewave.evaluate import evaluate
from squarewave.remote import BASES, RemoteOracle

N_SUBS = 150
CHUNK = 16


def gradient_check(ro: RemoteOracle, seq: str, rng: np.random.Generator) -> dict:
    g = ro.score_and_grad([seq])
    grad, loss0 = g["grad"][0], g["loss"][0]
    base = np.array([BASES.index(c) for c in seq])
    pos = rng.choice(DESIGN_BP, size=N_SUBS, replace=False)
    alt = (base[pos] + rng.integers(1, 4, size=N_SUBS)) % 4
    muts = []
    for p, a in zip(pos, alt):
        s = list(seq)
        s[p] = BASES[a]
        muts.append("".join(s))
    act = np.concatenate([ro.score(muts[i : i + CHUNK])["loss"] for i in range(0, N_SUBS, CHUNK)]) - loss0
    pred = grad[pos, alt] - grad[pos, base[pos]]
    return {
        "pearson": float(np.corrcoef(pred, act)[0, 1]),
        "spearman": float(spearmanr(pred, act)[0]),
        "sign_agreement": float((np.sign(pred) == np.sign(act)).mean()),
        "mean_abs_actual_dloss": float(np.abs(act).mean()),
        "grad_finite": bool(g["finite"]),
    }


def main() -> None:
    ro = RemoteOracle()
    starts = {f"{kind}_s{s}": make_start(kind, s) for kind in ("uniform", "markov") for s in range(4)}
    rows = dict(zip(starts, evaluate(ro, list(starts.values()))))
    keys = ["loss", "auroc", "heldout_auroc", "gc", "cpg_obs_exp", "dinuc_l1_vs_mm39"]
    print("%-12s" % "start" + " ".join("%12s" % k[:12] for k in keys))
    for name, r in rows.items():
        print("%-12s" % name + " ".join("%12.3f" % r[k] for k in keys))

    grads = {}
    for kind in ("uniform", "markov"):
        for s in range(3):
            grads[f"{kind}_s{s}"] = gradient_check(ro, starts[f"{kind}_s{s}"], np.random.default_rng(50 + s))
            print(kind, s, {k: round(v, 3) if isinstance(v, float) else v for k, v in grads[f"{kind}_s{s}"].items()}, flush=True)
    for kind in ("uniform", "markov"):
        m = {k: float(np.mean([grads[f"{kind}_s{s}"][k] for s in range(3)])) for k in ("pearson", "spearman", "sign_agreement")}
        print(f"mean over 3 {kind} starts:", {k: round(v, 3) for k, v in m.items()})
    Path("results/markov_start_check.json").write_text(json.dumps({"starts": rows, "gradient": grads}, indent=1))
    print(f"est. GPU cost ${ro.usd_estimate:.2f}")


if __name__ == "__main__":
    main()
