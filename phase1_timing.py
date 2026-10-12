"""Warm forward-pass latency for the deployed Enformer/Borzoi apps (budget estimate)."""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from squarewave import scoring as S  # noqa: E402

from proto_tools.modal.client import dispatch_to_modal  # noqa: E402
from proto_tools.tools.sequence_scoring.borzoi.borzoi_ensemble import BorzoiEnsembleConfig  # noqa: E402
from proto_tools.tools.sequence_scoring.borzoi.borzoi_prediction import BorzoiInput  # noqa: E402
from proto_tools.tools.sequence_scoring.enformer.enformer_prediction import EnformerConfig, EnformerInput  # noqa: E402

up, down = S.load_flanks()
rng = np.random.default_rng(1)
n = 19_584


def rand_design() -> str:
    return "".join(rng.choice(list("ACGT"), size=n))


for batch in (1, 4):
    designs = [rand_design() for _ in range(batch)]
    ew = [S.assemble(d, up, down, S.ENF_CTX, S.ENF_UP) for d in designs]
    bw = [S.assemble(d, up, down, S.BOR_CTX, S.BOR_UP) for d in designs]
    for rep in range(2):
        t = time.time()
        dispatch_to_modal(
            "enformer-prediction",
            EnformerInput(sequences=ew),
            EnformerConfig(species="mouse", output_tracks=[11], batch_size=batch),
            environment=os.environ.get("MODAL_ENVIRONMENT"),
        )
        te = time.time() - t
        t = time.time()
        dispatch_to_modal(
            "borzoi-ensemble",
            BorzoiInput(sequences=bw),
            BorzoiEnsembleConfig(species="mouse", output_tracks=[741], avg_output_tracks=False, batch_size=batch),
            environment=os.environ.get("MODAL_ENVIRONMENT"),
        )
        tb = time.time() - t
        print(f"batch={batch} rep={rep}: enformer {te:.1f}s ({te / batch:.2f}/seq) | borzoi {tb:.1f}s ({tb / batch:.2f}/seq)", flush=True)
