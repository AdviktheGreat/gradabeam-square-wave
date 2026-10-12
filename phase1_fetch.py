"""Phase 1a: score the published long-square-wave design (plus controls) with the deployed Enformer/Borzoi apps.

Run with the proto-tools interpreter:
    python phase1_fetch.py   # needs the proto-tools package and its Enformer/Borzoi apps deployed on Modal
"""

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

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

ENF_TRACKS = [9, 10, 11, 12, 13]
BOR_TRACKS = [739, 740, 741, 742, 743]


def published_long_wave() -> str:
    return S.load_published_long_wave()


def main() -> None:
    up, down = S.load_flanks()
    native = (S.DATA / "mm39_chrX_native_region.txt").read_text().strip()
    pub = published_long_wave()
    n = len(pub)
    rng = np.random.default_rng(0)
    rand = "".join(rng.choice(list("ACGT"), size=n))
    designs = {"published_long_wave": pub, "uniform_random": rand, "native_region_prefix": native[:n]}

    enf_windows = [S.assemble(d, up, down, S.ENF_CTX, S.ENF_UP) for d in designs.values()]
    bor_windows = [S.assemble(d, up, down, S.BOR_CTX, S.BOR_UP) for d in designs.values()]

    t = time.time()
    enf = dispatch_to_modal(
        "enformer-prediction",
        EnformerInput(sequences=enf_windows),
        EnformerConfig(species="mouse", output_tracks=ENF_TRACKS, batch_size=1),
        environment=os.environ.get("MODAL_ENVIRONMENT"),
    )
    t_enf = time.time() - t
    print(f"enformer: {t_enf:.1f}s wall, tool-reported {enf.execution_time}s", flush=True)
    r0 = enf.results[0]
    print("enformer geometry:", r0.context_start, r0.context_end, r0.output_start, r0.output_end, r0.output_resolution)

    t = time.time()
    bor = dispatch_to_modal(
        "borzoi-ensemble",
        BorzoiInput(sequences=bor_windows),
        BorzoiEnsembleConfig(species="mouse", output_tracks=BOR_TRACKS, avg_output_tracks=False, batch_size=1),
        environment=os.environ.get("MODAL_ENVIRONMENT"),
    )
    t_bor = time.time() - t
    print(f"borzoi: {t_bor:.1f}s wall, tool-reported {bor.execution_time}s", flush=True)
    b0 = bor.results[0]
    print("borzoi geometry:", b0.context_start, b0.context_end, b0.output_start, b0.output_end, b0.output_resolution)

    np.savez_compressed(
        OUT / "phase1_raw.npz",
        names=np.array(list(designs)),
        enf=np.array([np.array(r.prediction) for r in enf.results]),
        bor=np.array([np.array(r.predictions) for r in bor.results]),
        enf_tracks=np.array(ENF_TRACKS),
        bor_tracks=np.array(BOR_TRACKS),
        timing=np.array([t_enf, t_bor]),
    )
    print("saved", OUT / "phase1_raw.npz")


if __name__ == "__main__":
    main()
