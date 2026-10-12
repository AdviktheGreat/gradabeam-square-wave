"""Client for the deployed gradabeam-squarewave Modal app.

The paper loss is f = (f_enformer + f_borzoi) / 2 and is separable, so the two services are called concurrently
and their outputs summed. Thread-safe: GPU-seconds are accumulated under a lock so parallel runs share one ledger.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import modal
import numpy as np

from squarewave.oracle_server import APP_NAME

ENV = os.environ.get("MODAL_ENVIRONMENT")  # None selects the active Modal profile's default environment
BASES = "ACGT"
# Modal's published H100 rate (USD per GPU-second) for budget tracking; actual billing also covers idle and start-up.
H100_USD_PER_S = 3.95 / 3600


class RemoteOracle:
    def __init__(self, open_bp: int = 1664, closed_bp: int = 1664, env: str = ENV):
        self.open_bp, self.closed_bp = open_bp, closed_bp
        self.enf = modal.Cls.from_name(APP_NAME, "EnformerOracle", environment_name=env)()
        self.bor = modal.Cls.from_name(APP_NAME, "BorzoiOracle", environment_name=env)()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2)
        self.gpu_s = {"enformer": 0.0, "borzoi": 0.0}
        self.calls = {"score": 0, "grad": 0}

    def _book(self, key: str, enf: dict, bor: dict, n: int) -> None:
        with self._lock:
            self.gpu_s["enformer"] += enf["gpu_s"]
            self.gpu_s["borzoi"] += bor["gpu_s"]
            self.calls[key] += n

    @property
    def usd_estimate(self) -> float:
        return sum(self.gpu_s.values()) * H100_USD_PER_S

    @staticmethod
    def _retry(fn, designs, kw, attempts: int = 8):
        for i in range(attempts):
            try:
                return fn.remote(designs, **kw)
            except Exception:  # transient Modal/network failures; the final attempt re-raises
                if i == attempts - 1:
                    raise
                time.sleep(min(2.0 * 2**i, 60.0))

    def _both(self, method: str, designs: list[str], **kw) -> tuple[dict, dict]:
        kw = {"open_bp": self.open_bp, "closed_bp": self.closed_bp, **kw}
        fe = self._pool.submit(self._retry, getattr(self.enf, method), designs, kw)
        fb = self._pool.submit(self._retry, getattr(self.bor, method), designs, kw)
        return fe.result(), fb.result()

    def score(self, designs: list[str], return_raw: bool = False) -> dict:
        e, b = self._both("score", designs, return_raw=return_raw)
        self._book("score", e, b, len(designs))
        out = {
            "loss": 0.5 * (np.array(e["f"]) + np.array(b["f"])),
            "f_enformer": np.array(e["f"]),
            "f_borzoi": np.array(b["f"]),
        }
        if return_raw:
            out["raw_enformer"] = np.frombuffer(e["raw"], dtype=np.float32).reshape(len(designs), -1)
            out["raw_borzoi"] = np.frombuffer(b["raw"], dtype=np.float32).reshape(len(designs), 4, -1)
        return out

    def score_and_grad(self, designs: list[str]) -> dict:
        e, b = self._both("score_and_grad", designs)
        self._book("grad", e, b, len(designs))
        n = len(designs)
        ge = np.frombuffer(e["grad"], dtype=np.float32).reshape(n, -1, 4)
        gb = np.frombuffer(b["grad"], dtype=np.float32).reshape(n, -1, 4)
        return {
            "loss": 0.5 * (np.array(e["f"]) + np.array(b["f"])),
            "grad": 0.5 * (ge + gb),
            "grad_enformer": ge,
            "grad_borzoi": gb,
            "finite": b["finite"],
            "peak_gb_borzoi": b["peak_gb"],
            "loss_scale": b["scale"],
        }

    def raw_tracks(self, designs: list[str], enf_tracks: list[int], bor_tracks: list[int]) -> dict:
        """Raw predictions for extra mouse-head tracks: enformer (B, T, 896), borzoi (B, 4, T, 6144)."""
        fe = self._pool.submit(self._retry, self.enf.raw_tracks, designs, {"tracks": enf_tracks})
        fb = self._pool.submit(self._retry, self.bor.raw_tracks, designs, {"tracks": bor_tracks})
        e, b = fe.result(), fb.result()
        n = len(designs)
        return {
            "enformer": np.frombuffer(e["raw"], dtype=np.float32).reshape(n, len(enf_tracks), -1),
            "borzoi": np.frombuffer(b["raw"], dtype=np.float32).reshape(n, 4, len(bor_tracks), -1),
        }
