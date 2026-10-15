"""Figure for the Markov-start follow-up: loss and AUROC vs budget, and how genome-like the final designs are.

Reads only saved results (no oracle calls):  python make_markov_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
N_CHUNKS = 153
TOK = np.array([1, 2, 3, 6, 9, 12, 15, 30])
# Read by eye from Extended Data Fig. 10c (mean over six patterns), same as make_figure.py.
EVO2_PAPER = np.array([0.51, 0.70, 0.735, 0.76, 0.84, 0.855, 0.88, 0.955])

COLORS = {"markov": "tab:blue", "default": "tab:red", "main": "tab:orange"}
NAMES = {"markov": "Markov-o5 start, default cfg (4 seeds)", "default": "uniform start, default cfg (2 seeds)", "main": "uniform start, tuned cfg (4 seeds)"}


def arm(run: str) -> str:
    return run.split("_")[0]


def main() -> None:
    rows = json.loads((R / "analysis_markov.json").read_text()) + json.loads((R / "analysis_main.json").read_text())
    base = json.loads((R / "baselines.json").read_text())
    ub = [json.loads(f.read_text()) for f in sorted((R / "uniform_beam").glob("*.json"))]
    ub_rows = json.loads((R / "ubeam_eval.json").read_text())
    evo2 = base["evo2_published_long_wave"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    seen = set()
    for d in sorted(R.glob("runs/*")):
        if arm(d.name) not in COLORS or d.name.startswith("pilot") or not (d / "trace.npz").exists():
            continue
        t = np.load(d / "trace.npz")["trace"]
        a = arm(d.name)
        ax.plot(t[:, 0], t[:, 2], color=COLORS[a], lw=1.1, alpha=0.5 if a == "main" else 0.9, label=NAMES[a] if a not in seen else None)
        seen.add(a)
    ax.axhline(evo2["loss"], color="k", ls="--", lw=1, label=f"Evo 2 published design ({evo2['loss']:.1f}, rescored)")
    ax.set_ylim(110, 200)
    ax.set_xlabel("forward-equivalents (1 score = 1, 1 gradient = 2.1)")
    ax.set_ylabel("best loss so far (lower is better)")
    ax.set_title("Loss vs oracle budget", fontsize=10, loc="left")
    ax.legend(fontsize=7)

    ax = axes[1]
    seen = set()
    for run in sorted({r["run"] for r in rows if arm(r["run"]) in COLORS}):
        rr = [r for r in rows if r["run"] == run]
        a = arm(run)
        ax.plot([r["fe"] for r in rr], [r["auroc"] for r in rr], "o-", ms=3, lw=1, color=COLORS[a], alpha=0.5 if a == "main" else 0.9, label=NAMES[a] if a not in seen else None)
        seen.add(a)
    ax.plot(TOK * N_CHUNKS, EVO2_PAPER, "s-", color="k", ms=4, lw=1.2, label="Evo 2 7B (paper, read off figure)")
    ax.scatter([12852], [evo2["auroc"]], color="k", marker="*", s=120, zorder=5, label=f"Evo 2 published long wave ({evo2['auroc']:.3f})")
    ax.scatter([u["n_scored"] for u in ub], [r["auroc"] for r in ub_rows], color="tab:gray", marker="x", s=40, label="uniform-proposal beam search")
    ax.set_xscale("log")
    ax.set_xlabel("scored candidates / forward-equivalents (log)")
    ax.set_ylabel("AUROC (mean of Enformer, Borzoi)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Design success vs oracle budget", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    seen = set()
    for r in [r for r in rows if r["fe"] == 13000 and arm(r["run"]) in COLORS]:
        a = arm(r["run"])
        ax.scatter(r["auroc"], r["dinuc_l1_vs_mm39"], color=COLORS[a], s=45, alpha=0.9, label=NAMES[a] if a not in seen else None)
        seen.add(a)
        if r["run"] == "markov_default_s1":
            ax.annotate("Markov seed 1\n(68% of bases edited)", (r["auroc"], r["dinuc_l1_vs_mm39"]), textcoords="offset points", xytext=(-8, -34), fontsize=7, ha="center", arrowprops={"arrowstyle": "-", "lw": 0.6})
    ax.scatter([r["auroc"] for r in ub_rows], [r["dinuc_l1_vs_mm39"] for r in ub_rows], color="tab:gray", marker="x", s=40, label="uniform-proposal beam search")
    ax.scatter([evo2["auroc"]], [evo2["dinuc_l1_vs_mm39"]], color="k", marker="*", s=120, label="Evo 2 published design")
    ax.set_xlabel("AUROC")
    ax.set_ylabel("dinucleotide distance to native mm39 (L1; 0 = genome-like)")
    ax.set_title("Final designs: success vs genome-likeness", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    fig.savefig(R / "markov_start.png", dpi=130)
    print("saved", R / "markov_start.png")


if __name__ == "__main__":
    main()
