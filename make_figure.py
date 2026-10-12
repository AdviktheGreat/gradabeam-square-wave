"""Summary figure: GradaBeam vs Evo 2 / uniform-proposal beam search on the long square wave."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from squarewave import scoring as S
from squarewave.remote import RemoteOracle

R = Path("results")
N_CHUNKS = 153
# Evo 2 and uniform-proposal AUROC vs tokens/bp, read by eye from Extended Data Fig. 10c (mean over six patterns).
TOK = np.array([1, 2, 3, 6, 9, 12, 15, 30])
EVO2_PAPER = np.array([0.51, 0.70, 0.735, 0.76, 0.84, 0.855, 0.88, 0.955])
UNIF_PAPER = np.array([0.48, 0.645, 0.66, 0.75, 0.775, 0.79, 0.835, 0.835])


def profiles(ro: RemoteOracle, seqs: dict[str, str]) -> dict:
    n = len(next(iter(seqs.values())))
    target = S.wave_pattern_bp(n, S.LONG_WAVE_OPEN_BP, S.LONG_WAVE_CLOSED_BP)
    out = ro.score(list(seqs.values()), return_raw=True)
    res = {}
    for i, name in enumerate(seqs):
        d = S.design_loss(out["raw_enformer"][i], out["raw_borzoi"][i], target)
        res[name] = (d["profile_enformer"][: n // S.ENF_BIN], d["profile_borzoi"][: n // S.BOR_BIN])
    return res, target


def main() -> None:
    pub = S.load_published_long_wave()
    seqs = {
        "Evo 2 published design": pub,
        "GradaBeam (default cfg, seed 0), 13k fe": (R / "runs/default_s0/best.txt").read_text(),
        "Uniform-proposal beam search, 84 tok/bp": json.loads((R / "uniform_beam/ubeam_kp2_k42_s1.json").read_text())["seq"],
    }
    ro = RemoteOracle()
    prof, target = profiles(ro, seqs)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2)
    bins_e = np.arange(len(next(iter(prof.values()))[0])) * 128 / 1000
    bins_b = np.arange(len(next(iter(prof.values()))[1])) * 32 / 1000
    tgt_x = np.arange(len(target)) / 1000
    for k, (name, (pe, pb)) in enumerate(prof.items()):
        ax = fig.add_subplot(gs[k, 0])
        ax.fill_between(tgt_x, target, step="mid", color="0.88", label="target (open = 1)")
        ax.plot(bins_e, pe, color="tab:green", lw=1.2, label="Enformer (max-normalised)")
        ax.plot(bins_b, pb, color="tab:blue", lw=1.0, label="Borzoi mean - std")
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_xlim(0, tgt_x[-1])
        ax.set_ylabel("normalised prediction")
        if k == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_xlabel("position in design (kb)")

    ax = fig.add_subplot(gs[0, 1])
    colors = {"main": "tab:orange", "default": "tab:red"}
    for run_dir in sorted((R / "runs").glob("main_s*")) + sorted((R / "runs").glob("default_s*")):
        t = np.load(run_dir / "trace.npz")["trace"]
        kind = run_dir.name.split("_")[0]
        ax.plot(t[:, 0], t[:, 2], color=colors[kind], lw=1, alpha=0.8, label="GradaBeam " + ("tuned (pilot)" if kind == "main" else "default") if run_dir.name.endswith("s0") else None)
    for f in sorted((R / "uniform_beam").glob("*.json")):
        d = json.loads(f.read_text())
        ax.scatter(d["n_scored"], d["final_loss"], color="tab:gray", marker="x", s=40, label="uniform beam (final)" if f.name.endswith("kp2_k15_s0.json") else None)
    ax.axhline(135.71, color="k", ls="--", lw=1, label="Evo 2 published design (135.7)")
    ax.set_ylim(125, 200)
    ax.set_xlabel("forward-equivalents (1 score = 1, 1 gradient = 2.1)")
    ax.set_ylabel("best loss so far (lower is better)")
    ax.legend(fontsize=7)
    ax.set_title("Loss vs oracle budget", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[1, 1])
    rows = json.loads((R / "analysis_main.json").read_text())
    for run in sorted({r["run"] for r in rows}):
        rr = [r for r in rows if r["run"] == run]
        kind = run.split("_")[0]
        ax.plot([r["fe"] for r in rr], [r["auroc"] for r in rr], "o-", ms=3, lw=1, color=colors[kind], alpha=0.8)
    ax.plot(TOK * N_CHUNKS, EVO2_PAPER, "s-", color="k", ms=4, lw=1.2, label="Evo 2 7B (paper, avg of 6 patterns)")
    ax.plot(TOK * N_CHUNKS, UNIF_PAPER, "^-", color="tab:blue", ms=4, lw=1.2, label="uniform proposal (paper, avg of 6)")
    ax.scatter([12852], [0.998], color="k", marker="*", s=120, zorder=5, label="Evo 2 published long wave (0.998)")
    ub = [json.loads(f.read_text()) for f in sorted((R / "uniform_beam").glob("*.json"))]
    ub_rows = json.loads((R / "ubeam_eval.json").read_text())
    ax.scatter([u["n_scored"] for u in ub], [r["auroc"] for r in ub_rows], color="tab:gray", marker="x", s=40, label="uniform beam (this pipeline)")
    ax.set_xscale("log")
    ax.set_xlabel("scored candidates / forward-equivalents (log)")
    ax.set_ylabel("AUROC (mean of Enformer, Borzoi)")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Design success vs oracle budget", fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(R / "summary.png", dpi=130)
    print("saved", R / "summary.png")

    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    base = json.loads((R / "baselines.json").read_text())
    labels, op, cl = [], [], []
    fin = {r["run"]: r for r in rows if r["fe"] == 13000}
    for label, r in [("Evo 2\npublished", base["evo2_published_long_wave"]), ("native\nmm39", base["native_mm39_region"]), ("uniform\nrandom", base["uniform_random"])] + [
        (f"{k}", v) for k, v in fin.items()
    ] + [(f"ubeam\n{json.loads(f.read_text())['tok_per_bp']}tok s{json.loads(f.read_text())['seed']}", r) for f, r in zip(sorted((R / 'uniform_beam').glob('*.json')), ub_rows)]:
        labels.append(label)
        op.append(r["gc_open"])
        cl.append(r["gc_closed"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, op, 0.4, label="GC, open segments", color="tab:red")
    ax.bar(x + 0.2, cl, 0.4, label="GC, closed segments", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.set_ylim(0.3, 0.56)
    ax.set_ylabel("GC fraction")
    ax.legend(fontsize=7)
    ax.set_title("Sequence composition follows the wave only in the Evo 2 design", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[0, 1])
    keys = [("auroc", "AUROC (optimised tracks)"), ("heldout_auroc", "AUROC (held-out ES tracks)"), ("icc", "ICC (ensemble agreement)")]
    labs = ["Evo 2\npublished", "GradaBeam\ndefault", "GradaBeam\ntuned", "uniform beam\n84 tok/bp"]
    vals = {
        "auroc": [base["evo2_published_long_wave"]["auroc"], np.mean([fin[k]["auroc"] for k in fin if k.startswith("default")]), np.mean([fin[k]["auroc"] for k in fin if k.startswith("main")]), np.mean([r["auroc"] for u, r in zip(ub, ub_rows) if u["k"] == 42])],
        "heldout_auroc": [base["evo2_published_long_wave"]["heldout_auroc"], np.mean([fin[k]["heldout_auroc"] for k in fin if k.startswith("default")]), np.mean([fin[k]["heldout_auroc"] for k in fin if k.startswith("main")]), np.mean([r["heldout_auroc"] for u, r in zip(ub, ub_rows) if u["k"] == 42])],
        "icc": [base["evo2_published_long_wave"]["icc"], np.mean([fin[k]["icc"] for k in fin if k.startswith("default")]), np.mean([fin[k]["icc"] for k in fin if k.startswith("main")]), np.mean([r["icc"] for u, r in zip(ub, ub_rows) if u["k"] == 42])],
    }
    x = np.arange(len(labs))
    for j, (k, title) in enumerate(keys):
        ax.bar(x + (j - 1) * 0.27, vals[k], 0.27, label=title)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(0.6, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Means over seeds at ~13k scored candidates", fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(R / "checks.png", dpi=130)
    print("saved", R / "checks.png")


if __name__ == "__main__":
    main()
