"""The long-square-wave oracle as a torch Module, so Ledidi (jmschrei/ledidi) can edit sequences against it.

Ledidi wants a frozen differentiable ``model(x) -> y_hat`` on one-hot batches of shape (B, 4, L). The real oracle
is two remote GPU services, so the Module's forward is a custom autograd Function: it decodes the hard one-hot
samples to strings, calls the Modal app once per design (concurrently), and returns the per-design paper loss as
y_hat with shape (B, 1). Backward feeds d(loss)/d(one-hot) from the remote backward pass to Ledidi's
straight-through estimator. Ledidi's output_loss is then simply the batch mean of y_hat.

Budget accounting and the best-so-far bookkeeping match gb_oracle.SquareWaveOracle: every scored design counts,
whichever optimizer produced it, so runs from the two optimizers are comparable at equal forward-equivalents.
"""

import collections
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import torch

from squarewave.budget import GRAD_FE, BudgetExhausted
from squarewave.remote import BASES, RemoteOracle

_LUT = np.frombuffer(BASES.encode(), dtype=np.uint8)  # index -> ASCII
_IDX = np.full(256, -1, dtype=np.int64)  # ASCII -> index
for _i, _c in enumerate(BASES):
    _IDX[ord(_c)] = _i


def seq_to_idx(seq: str) -> np.ndarray:
    idx = _IDX[np.frombuffer(seq.encode(), dtype=np.uint8)]
    assert idx.min() >= 0, "sequence must be ACGT only"
    return idx


def to_onehot(seq: str) -> torch.Tensor:
    """(1, 4, L) float tensor, channel order ACGT as Ledidi expects."""
    idx = seq_to_idx(seq)
    x = np.zeros((1, 4, len(seq)), dtype=np.float32)
    x[0, idx, np.arange(len(seq))] = 1.0
    return torch.from_numpy(x)


def to_strings(x: torch.Tensor) -> list[str]:
    """Decode a (B, 4, L) batch of (hard, possibly straight-through) one-hots."""
    idx = x.detach().argmax(dim=1).numpy()
    return [_LUT[row].tobytes().decode() for row in idx]


class MeanLoss(torch.nn.Module):
    """Ledidi output_loss: y_hat already holds each design's paper loss, so just average over the batch."""

    def forward(self, y_hat: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
        return y_hat.mean()


class _OracleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, oracle: "LedidiOracle") -> torch.Tensor:
        seqs = to_strings(x)
        if ctx.needs_input_grad[0]:
            loss, grad = oracle.loss_and_grad(seqs)
            ctx.save_for_backward(torch.from_numpy(grad))
        else:
            loss = oracle.loss_only(seqs)
        return torch.from_numpy(loss.astype(np.float32)).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (grad,) = ctx.saved_tensors
        return grad_out.reshape(-1, 1, 1) * grad, None


class LedidiOracle(torch.nn.Module):
    def __init__(
        self,
        remote: RemoteOracle,
        template: str,
        budget_fe: float,
        concurrency: int = 4,
        milestone_fe: float = 250.0,
        on_progress: Callable[["LedidiOracle"], None] | None = None,
        progress_every_s: float = 60.0,
    ):
        super().__init__()
        self.remote = remote
        self.budget_fe = budget_fe
        self.milestone_fe = milestone_fe
        self._template_idx = seq_to_idx(template)
        self._pool = ThreadPoolExecutor(max_workers=concurrency)
        self._on_progress = on_progress
        self._progress_every_s = progress_every_s
        self._last_progress = time.time()
        self.n_fwd = 0
        self.n_grad = 0
        self.n_bad_grad = 0
        self.best_loss = float("inf")
        self.best_seq = ""
        self.t0 = time.time()
        self.trace: list[tuple[float, float, float, float]] = []  # (fe, design loss, best loss, wall s)
        self.iters: list[tuple[float, float, float, float]] = []  # (fe, batch mean loss, batch min loss, mean edits)
        self.grad_scales: collections.Counter = collections.Counter()
        self.milestones: dict[int, tuple[float, str]] = {}
        self._next_milestone = milestone_fe
        self._lock = threading.Lock()

    @property
    def fe(self) -> float:
        return self.n_fwd + GRAD_FE * self.n_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _OracleFn.apply(x, self)

    def _observe(self, loss: float, seq: str) -> None:
        if loss < self.best_loss:
            self.best_loss, self.best_seq = float(loss), seq
        self.trace.append((self.fe, float(loss), self.best_loss, time.time() - self.t0))
        while self.fe >= self._next_milestone:
            self.milestones[int(self._next_milestone)] = (self.best_loss, self.best_seq)
            self._next_milestone += self.milestone_fe

    def _log_batch(self, losses: np.ndarray, seqs: list[str]) -> None:
        edits = np.mean([(seq_to_idx(s) != self._template_idx).sum() for s in seqs])
        self.iters.append((self.fe, float(losses.mean()), float(losses.min()), float(edits)))
        if self._on_progress is not None and time.time() - self._last_progress > self._progress_every_s:
            self._last_progress = time.time()
            self._on_progress(self)

    def loss_only(self, seqs: list[str]) -> np.ndarray:
        if self.fe + len(seqs) > self.budget_fe:
            raise BudgetExhausted
        out = self.remote.score(list(seqs))
        with self._lock:
            for loss, seq in zip(out["loss"], seqs):
                self.n_fwd += 1
                self._observe(loss, seq)
        return np.asarray(out["loss"])

    def loss_and_grad(self, seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Per-design loss (B,) and d(loss)/d(one-hot) in Ledidi's layout (B, 4, L)."""
        if self.fe + GRAD_FE * len(seqs) > self.budget_fe:
            raise BudgetExhausted
        outs = list(self._pool.map(lambda s: self.remote.score_and_grad([s]), seqs))
        losses, grads = [], []
        with self._lock:
            for seq, out in zip(seqs, outs):
                g = np.array(out["grad"][0])
                if not (out["finite"] and np.isfinite(g).all()):
                    # A zero gradient costs this design's step but keeps a long run alive.
                    self.n_bad_grad += 1
                    g = np.zeros_like(g)
                self.n_grad += 1
                self.grad_scales[out["loss_scale"]] += 1
                self._observe(float(out["loss"][0]), seq)
                losses.append(float(out["loss"][0]))
                grads.append(g.T.copy())  # (L, 4) -> (4, L)
            losses = np.array(losses)
            self._log_batch(losses, seqs)
        return losses, np.stack(grads).astype(np.float32)
