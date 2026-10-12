"""Phase 2b: validate the custom Modal oracle against the Phase 1 numbers and check gradients.

Run:  PYTHONPATH=. python phase2_validate.py
Checks:
  1. forward loss of the published long wave vs Phase 1 (Enformer 38.82, Borzoi 232.57, loss 135.70)
  2. forward loss of uniform random vs Phase 1 (195.84)
  3. gradient vs finite differences (first-order change from single-base substitutions)
  4. latency, peak memory
"""

import time
from pathlib import Path

import numpy as np

from squarewave import scoring as S
from squarewave.remote import BASES, RemoteOracle


def main() -> None:
    pub = S.load_published_long_wave()
    n = len(pub)
    rng = np.random.default_rng(0)
    rand = "".join(rng.choice(list(BASES), size=n))
    ro = RemoteOracle()

    t = time.time()
    r = ro.score([pub, rand], return_raw=True)
    print(f"score (cold): {time.time() - t:.1f}s")
    print("published: loss %.2f  E %.2f  B %.2f   (phase1: 135.70 / 38.82 / 232.57)" % (r["loss"][0], r["f_enformer"][0], r["f_borzoi"][0]))
    print("random:    loss %.2f  E %.2f  B %.2f   (phase1: 195.84)" % (r["loss"][1], r["f_enformer"][1], r["f_borzoi"][1]))

    target = S.wave_pattern_bp(n, S.LONG_WAVE_OPEN_BP, S.LONG_WAVE_CLOSED_BP)
    for i, name in enumerate(["published", "random"]):
        d = S.design_loss(r["raw_enformer"][i], r["raw_borzoi"][i], target)
        a = S.design_auroc(d["profile_enformer"], d["profile_borzoi"], target)
        print(f"  numpy re-score {name}: loss {d['loss']:.2f} auroc {a['auroc']:.3f}")

    t = time.time()
    r2 = ro.score([pub])
    print(f"score warm x1: {time.time() - t:.2f}s  loss {r2['loss'][0]:.2f}")
    t = time.time()
    r2 = ro.score([pub] * 4)
    print(f"score warm x4: {time.time() - t:.2f}s")

    t = time.time()
    g = ro.score_and_grad([rand])
    print(f"grad cold: {time.time() - t:.1f}s  loss {g['loss'][0]:.2f} finite={g['finite']} borzoi peak {g['peak_gb_borzoi']:.1f} GB")
    t = time.time()
    g = ro.score_and_grad([rand])
    print(f"grad warm x1: {time.time() - t:.2f}s")

    grad = g["grad"][0]  # (n, 4)
    gb, ge = g["grad_borzoi"][0], g["grad_enformer"][0]
    print("grad norms: enformer %.3e borzoi %.3e; frac of design positions with nonzero grad: %.3f" % (np.linalg.norm(ge), np.linalg.norm(gb), (np.abs(grad).sum(1) > 0).mean()))

    # first-order prediction vs actual loss change for single-base substitutions
    base_idx = np.array(["ACGT".index(c) for c in rand])
    positions = rng.choice(n, size=24, replace=False)
    muts, pred = [], []
    for p in positions:
        alt = (base_idx[p] + rng.integers(1, 4)) % 4
        s = list(rand)
        s[p] = BASES[alt]
        muts.append("".join(s))
        pred.append(grad[p, alt] - grad[p, base_idx[p]])
    act = ro.score(muts)["loss"] - g["loss"][0]
    pred = np.array(pred)
    print("single-substitution loss change: corr(pred, actual) = %.3f ; |pred| mean %.3e, |actual| mean %.3e" % (np.corrcoef(pred, act)[0, 1], np.abs(pred).mean(), np.abs(act).mean()))
    print("sign agreement: %.2f" % (np.sign(pred) == np.sign(act)).mean())
    print(f"est. GPU-seconds enformer {ro.gpu_s['enformer']:.1f}, borzoi {ro.gpu_s['borzoi']:.1f}; est. cost ${ro.usd_estimate:.2f}")
    np.savez_compressed(Path(__file__).parent / "results/phase2_grad_check.npz", grad=grad, rand=np.array(rand), pred=pred, act=act)


if __name__ == "__main__":
    main()
