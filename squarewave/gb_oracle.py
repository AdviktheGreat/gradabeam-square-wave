"""GradaBeam oracle for the long-square-wave task, backed by the deployed Modal app.

Follows the oracles/template.py interface of gradabeam: ``__call__`` returns the paper loss (lower is better)
and ``get_tism`` returns first-order loss differences, which ``ModelWrapper`` negates.

Budget accounting is in forward-equivalents (fe): one scored candidate is 1 fe, one gradient call is GRAD_FE.
GRAD_FE is the measured ratio of H100 time for score_and_grad vs score on this oracle (0.45 s vs 0.21 s per call
across both services), not the optimizer's own tism_cost of 1.0, which would understate the gradient's price.
"""

import collections
import threading
import time

import numpy as np
import torch

from gradabeam import tism
from gradabeam.tism import TISMModelClass
from squarewave.remote import RemoteOracle

GRAD_FE = 2.1


class BudgetExhausted(Exception):
    pass


class SquareWaveOracle(TISMModelClass):
    vocab = ["A", "C", "G", "T"]
    vocab_array = np.array(vocab)
    vocab_to_idx = {c: i for i, c in enumerate(vocab)}

    def __init__(self, remote: RemoteOracle, budget_fe: float, milestone_fe: float = 250.0):
        self.remote = remote
        self.budget_fe = budget_fe
        self.milestone_fe = milestone_fe
        self.n_fwd = 0
        self.n_grad = 0
        self.best_loss = float("inf")
        self.best_seq = ""
        self.t0 = time.time()
        self.trace: list[tuple[float, float, float, float]] = []  # (fe, call loss, best loss, wall s)
        self.grad_scales: collections.Counter = collections.Counter()
        self.n_bad_grad = 0
        self.milestones: dict[int, tuple[float, str]] = {}
        self._next_milestone = milestone_fe
        self._lock = threading.Lock()

    @property
    def fe(self) -> float:
        return self.n_fwd + GRAD_FE * self.n_grad

    # --- interface expected by gradabeam.ada_utils.ModelWrapper ---
    def eval(self) -> "SquareWaveOracle":
        return self

    def parameters(self):
        return iter(())

    def _observe(self, losses: np.ndarray, seqs: list[str]) -> None:
        with self._lock:
            for loss, seq in zip(losses, seqs):
                if loss < self.best_loss:
                    self.best_loss, self.best_seq = float(loss), seq
                self.trace.append((self.fe, float(loss), self.best_loss, time.time() - self.t0))
            while self.fe >= self._next_milestone:
                self.milestones[int(self._next_milestone)] = (self.best_loss, self.best_seq)
                self._next_milestone += self.milestone_fe

    def __call__(self, seqs: list[str]) -> list[float]:
        if self.fe + len(seqs) > self.budget_fe:
            raise BudgetExhausted
        out = self.remote.score(list(seqs))
        with self._lock:
            self.n_fwd += len(seqs)
        self._observe(out["loss"], list(seqs))
        return [float(x) for x in out["loss"]]

    def tism_torch(self, x: str, idxs: list[int] | None = None) -> torch.Tensor:
        if self.fe + GRAD_FE > self.budget_fe:
            raise BudgetExhausted
        out = self.remote.score_and_grad([x])
        with self._lock:
            self.n_grad += 1
        self._observe(out["loss"], [x])
        with self._lock:
            self.grad_scales[out["loss_scale"]] += 1
        g = np.array(out["grad"][0])
        if not (out["finite"] and np.isfinite(g).all()):
            # Last resort so a long run is not lost: a zero gradient makes GradaBeam fall back to uniform proposals.
            with self._lock:
                self.n_bad_grad += 1
            g = np.zeros_like(g)
        grad = torch.from_numpy(g.T.copy())  # (4, L)
        if idxs is not None:
            grad = grad[:, idxs]
            x = "".join(x[i] for i in idxs)
        base = torch.tensor([self.vocab_to_idx[c] for c in x], dtype=torch.long)
        return tism.grad_torch_to_tism_torch(grad, base)
