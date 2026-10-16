"""Figure and statistics for the Ledidi follow-up: Ledidi vs GrAdaBeam, both from the same Markov-o5 starts.

Reads only saved results (no oracle calls):  python make_ledidi_figure.py
Writes results/ledidi.png and results/ledidi_stats.json.
"""

import itertools
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
SEEDS = range(4)
LEDIDI, GRADABEAM = "tab:green", "tab:blue"


def edited_fraction(run: str) -> float:
    start = (R / "runs" / run / "start.txt").read_text().strip()
    best = (R / "runs" / run / "best.txt").read_text().strip()
    return float(np.mean([a != b for a, b in zip(start, best)]))


def exact_perm_p(a: list[float], b: list[float]) -> dict:
    """Exact permutation test on the difference of means; returns both one-sided p-values and the two-sided p."""
    pooled = np.array(list(a) + list(b))
    obs = np.mean(a) - np.mean(b)
    diffs = []
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        diffs.append(pooled[mask].mean() - pooled[~mask].mean())
    diffs = np.array(diffs)
    return {
        "one_sided_a_lower": float((diffs <= obs + 1e-12).mean()),
        "one_sided_a_higher": float((diffs >= obs - 1e-12).mean()),
        "two_sided": float((np.abs(diffs) >= abs(obs) - 1e-12).mean()),
        "n_perm": len(diffs),
    }


def main() -> None:
    led = [r for r in json.loads((R / "analysis_ledidi.json").read_text())]
    gb = [r for r in json.loads((R / "analysis_markov.json").read_text())]
    pilot = [r for r in json.loads((R / "analysis_ledidi_pilot.json").read_text()) if r["fe"] == 5000]
    evo2 = json.loads((R / "baselines.json").read_text())["evo2_published_long_wave"]
    final = lambda rows, run: next(r for r in rows if r["run"] == run and r["fe"] == 13000)

    led_final = [final(led, f"ledidi_markov_s{s}") for s in SEEDS]
    gb_final = [final(gb, f"markov_default_s{s}") for s in SEEDS]
    stats = {
        "ledidi": {k: [r[k] for r in led_final] for k in ("loss", "auroc", "heldout_auroc", "icc", "cpg_obs_exp", "dinuc_l1_vs_mm39")},
        "gradabeam_markov": {k: [r[k] for r in gb_final] for k in ("loss", "auroc", "heldout_auroc", "icc", "cpg_obs_exp", "dinuc_l1_vs_mm39")},
        "edited_fraction_ledidi": [edited_fraction(f"ledidi_markov_s{s}") for s in SEEDS],
        "edited_fraction_gradabeam_markov": [edited_fraction(f"markov_default_s{s}") for s in SEEDS],
        "perm_loss_ledidi_vs_gradabeam": exact_perm_p([r["loss"] for r in led_final], [r["loss"] for r in gb_final]),
        "perm_auroc_ledidi_vs_gradabeam": exact_perm_p([r["auroc"] for r in led_final], [r["auroc"] for r in gb_final]),
        "evo2": {k: evo2[k] for k in ("loss", "auroc")},
    }
    (R / "ledidi_stats.json").write_text(json.dumps(stats, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for color, label, pattern in (
        (GRADABEAM, "GrAdaBeam, Markov start (4 seeds)", "markov_default_s*"),
        (LEDIDI, "Ledidi, Markov start (4 seeds)", "ledidi_markov_s*"),
    ):
        for i, d in enumerate(sorted((R / "runs").glob(pattern))):
            t = np.load(d / "trace.npz")["trace"]
            ax.plot(t[:, 0], t[:, 2], color=color, lw=1.1, alpha=0.9, label=label if i == 0 else None)
    ax.axhline(evo2["loss"], color="k", ls="--", lw=1, label=f"Evo 2 published design ({evo2['loss']:.1f}, rescored)")
    ax.set_xscale("log")
    ax.set_xlabel("forward-equivalents (1 score = 1, 1 gradient = 2.1; log scale)")
    ax.set_ylabel("best loss so far (lower is better)")
    ax.set_title("Loss vs oracle budget", fontsize=10, loc="left")
    ax.legend(fontsize=7)

    ax = axes[1]
    for rows, color, label, prefix in ((gb, GRADABEAM, "GrAdaBeam, Markov start", "markov_default"), (led, LEDIDI, "Ledidi, Markov start", "ledidi_markov")):
        for i, s in enumerate(SEEDS):
            rr = [r for r in rows if r["run"] == f"{prefix}_s{s}"]
            ax.plot([r["fe"] for r in rr], [r["auroc"] for r in rr], "o-", ms=3, lw=1, color=color, alpha=0.9, label=label if i == 0 else None)
    ax.plot(TOK * N_CHUNKS, EVO2_PAPER, "s-", color="k", ms=4, lw=1.2, label="Evo 2 7B (paper, read off figure)")
    ax.scatter([12852], [evo2["auroc"]], color="k", marker="*", s=120, zorder=5, label=f"Evo 2 published long wave ({evo2['auroc']:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("scored candidates / forward-equivalents (log)")
    ax.set_ylabel("AUROC (mean of Enformer, Borzoi)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Design success vs oracle budget", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    ax.scatter([r["loss"] for r in gb_final], [r["dinuc_l1_vs_mm39"] for r in gb_final], color=GRADABEAM, s=45, label="GrAdaBeam, Markov start, 13k fe")
    ax.scatter([r["loss"] for r in led_final], [r["dinuc_l1_vs_mm39"] for r in led_final], color=LEDIDI, s=45, label="Ledidi l=0.1 (default), Markov start, 13k fe")
    for r in pilot:
        l = r["run"].split("_l")[-1]
        if l == "0.1":
            continue
        ax.scatter([r["loss"]], [r["dinuc_l1_vs_mm39"]], facecolors="none", edgecolors=LEDIDI, s=55, marker="s")
        ax.annotate(f"l={l}", (r["loss"], r["dinuc_l1_vs_mm39"]), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.scatter([], [], facecolors="none", edgecolors=LEDIDI, marker="s", label="Ledidi pilot, other l (1 seed, 5k fe)")
    ax.scatter([evo2["loss"]], [evo2["dinuc_l1_vs_mm39"]], color="k", marker="*", s=120, label="Evo 2 published design")
    ax.set_xlabel("loss (lower is better)")
    ax.set_ylabel("dinucleotide distance to native mm39 (L1; 0 = genome-like)")
    ax.set_title("Final designs: loss vs genome-likeness", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="upper center")

    fig.tight_layout()
    fig.savefig(R / "ledidi.png", dpi=130)
    print("saved", R / "ledidi.png")
    print(json.dumps({k: v for k, v in stats.items() if k.startswith("perm")}, indent=1))


if __name__ == "__main__":
    main()
