"""Start sequences for the optimizer runs (uniform-random or order-5 Markov), shared by every run script."""

import numpy as np

from squarewave import markov
from squarewave import scoring as S
from squarewave.remote import BASES

DESIGN_BP = 19_584  # the published long-wave design length (SI Data 3)
MARKOV_MODEL = S.DATA / "markov_o5_mm39_autosomes.npz"


def make_start(kind: str, seed: int) -> str:
    """Start sequence for a run. The same seed number gives the same RNG stream for both kinds."""
    rng = np.random.default_rng(1000 + seed)
    if kind == "uniform":
        return "".join(rng.choice(list(BASES), size=DESIGN_BP))
    # Order-5 chain trained on mm39 autosomes, seeded with the last 5 bp of the native upstream flank so the junction is natural.
    probs = markov.load_probs(MARKOV_MODEL)
    up, _ = S.load_flanks()
    return markov.sample(probs, DESIGN_BP, up[-5:], rng)
