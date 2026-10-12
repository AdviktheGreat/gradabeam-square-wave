"""Modal oracle for the Evo 2 long-square-wave design task, with input gradients.

Reproduces the SI A.6 scoring pipeline (Enformer + Borzoi ensemble) as two independent GPU services.
The paper loss f = (f_enformer + f_borzoi) / 2 is separable, so the loss and its gradient with respect to the
design one-hot are the sums of the two services' contributions (done client-side, see remote.py).

Numerics deliberately match proto-tools' deployed predictors, which the Phase 1 validation used:
Enformer fp32 (EleutherAI/enformer-official-rough, mouse head, track 11); Borzoi mouse replicates 0-3
(johahi/borzoi-replicate-N-mouse, track 741) under default CUDA autocast.

Deploy:  modal deploy squarewave/oracle_server.py [--env <modal environment>]
The client (remote.py) reads the environment from $MODAL_ENVIRONMENT.
"""

import time
from pathlib import Path

import modal

APP_NAME = "gradabeam-squarewave"
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

ENF_CTX, ENF_UP, ENF_BIN, ENF_BINS = 196_608, 40_960, 128, 896
BOR_CTX, BOR_UP, BOR_BIN, BOR_BINS = 524_288, 163_840, 32, 6_144
ENF_TRACK, BOR_TRACK = 11, 741
BOR_REPLICATES = 4
BORZOI_MAX_CONTAINERS = 3
ENFORMER_MAX_CONTAINERS = 2
SCALEDOWN_S = 30

app = modal.App(APP_NAME)


def _download_enformer() -> None:
    from enformer_pytorch import from_pretrained

    from_pretrained("EleutherAI/enformer-official-rough")


def _download_borzoi() -> None:
    from borzoi_pytorch import Borzoi

    for r in range(BOR_REPLICATES):
        Borzoi.from_pretrained(f"johahi/borzoi-replicate-{r}-mouse")


def _with_data(image: modal.Image) -> modal.Image:
    return image.add_local_file(DATA / "mm39_chrX_flank_up163840.txt", "/data/up.txt").add_local_file(
        DATA / "mm39_chrX_flank_down400000.txt", "/data/down.txt"
    )


enformer_image = _with_data(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.1", "numpy", "enformer-pytorch==0.8.11", "transformers==4.36.0")
    .run_function(_download_enformer)
)
borzoi_image = _with_data(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.1", "numpy", "borzoi-pytorch==0.5.1")
    .run_function(_download_borzoi)
)


class _Common:
    """Sequence handling and loss pieces shared by both services. Methods import torch lazily."""

    ctx: int
    n_up: int
    bin_bp: int

    def _init_flanks(self) -> None:
        import numpy as np
        import torch

        up = Path("/data/up.txt").read_text().strip()
        down = Path("/data/down.txt").read_text().strip()
        lut = np.full(256, 4, dtype=np.int64)
        for i, c in enumerate("ACGT"):
            lut[ord(c)] = i
        self._lut = lut
        self._up_idx = torch.from_numpy(lut[np.frombuffer(up[len(up) - self.n_up :].encode(), dtype=np.uint8)]).cuda()
        self._down_full = torch.from_numpy(lut[np.frombuffer(down.encode(), dtype=np.uint8)]).cuda()

    def _design_idx(self, designs: list[str]):
        import numpy as np
        import torch

        lens = {len(d) for d in designs}
        assert len(lens) == 1, f"designs must share one length, got {lens}"
        arr = np.stack([self._lut[np.frombuffer(d.encode(), dtype=np.uint8)] for d in designs])
        assert arr.max() < 4, "designs must be ACGT only"
        return torch.from_numpy(arr).cuda()

    def _onehot_window(self, design_oh):
        """design_oh: (B, Ld, 4) float, may require grad -> (B, ctx, 4) with native flanks."""
        import torch
        import torch.nn.functional as F

        b, ld, _ = design_oh.shape
        n_down = self.ctx - self.n_up - ld
        assert 0 < n_down <= len(self._down_full)
        up = F.one_hot(self._up_idx, 4).float().unsqueeze(0).expand(b, -1, -1)
        down = F.one_hot(self._down_full[:n_down], 4).float().unsqueeze(0).expand(b, -1, -1)
        return torch.cat([up, design_oh, down], dim=1)

    @staticmethod
    def _target_bins(length_bp: int, bin_bp: int, open_bp: int, closed_bp: int):
        import torch

        pos = torch.arange(length_bp) % (open_bp + closed_bp)
        return (pos < open_bp).float().reshape(-1, bin_bp).mean(1).cuda()


def _pack(x) -> bytes:
    return x.detach().float().cpu().numpy().tobytes()


@app.cls(
    image=enformer_image,
    gpu="H100",
    max_containers=ENFORMER_MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_S,
    timeout=900,
)
@modal.concurrent(max_inputs=8, target_inputs=6)
class EnformerOracle(_Common):
    ctx, n_up, bin_bp = ENF_CTX, ENF_UP, ENF_BIN

    @modal.enter()
    def load(self) -> None:
        import threading

        import torch
        from enformer_pytorch import from_pretrained

        self._lock = threading.Lock()
        self.model = from_pretrained("EleutherAI/enformer-official-rough").cuda().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._init_flanks()

    def _raw(self, window_oh):
        return self.model(window_oh)["mouse"][:, :, ENF_TRACK]

    @staticmethod
    def _loss(raw, target):
        prof = raw / raw.max(dim=1, keepdim=True).values
        return (target.unsqueeze(0) - prof[:, : target.numel()]).abs().sum(1), prof

    @modal.method()
    def score(self, designs: list[str], open_bp: int = 1664, closed_bp: int = 1664, return_raw: bool = False,
              batch_size: int = 4) -> dict:
        import torch

        idx = self._design_idx(designs)
        ld = idx.shape[1]
        target = self._target_bins(ld, ENF_BIN, open_bp, closed_bp)
        losses, raws = [], []
        with self._lock, torch.inference_mode():
            t0 = time.time()
            for s in range(0, len(designs), batch_size):
                oh = torch.nn.functional.one_hot(idx[s : s + batch_size], 4).float()
                raw = self._raw(self._onehot_window(oh))
                loss, _ = self._loss(raw, target)
                losses.append(loss)
                raws.append(raw)
            torch.cuda.synchronize()
        out = {"f": torch.cat(losses).tolist(), "gpu_s": time.time() - t0}
        if return_raw:
            out["raw"] = _pack(torch.cat(raws))
        return out

    @modal.method()
    def score_and_grad(self, designs: list[str], open_bp: int = 1664, closed_bp: int = 1664) -> dict:
        import torch

        idx = self._design_idx(designs)
        target = self._target_bins(idx.shape[1], ENF_BIN, open_bp, closed_bp)
        with self._lock:
            t0 = time.time()
            x = torch.nn.functional.one_hot(idx, 4).float().requires_grad_(True)
            raw = self._raw(self._onehot_window(x))
            loss, _ = self._loss(raw, target)
            loss.sum().backward()
            torch.cuda.synchronize()
        return {"f": loss.detach().tolist(), "grad": _pack(x.grad), "gpu_s": time.time() - t0}

    @modal.method()
    def raw_tracks(self, designs: list[str], tracks: list[int], batch_size: int = 4) -> dict:
        """Raw mouse-head predictions for arbitrary tracks (held-out checks) -> (B, T, 896)."""
        import torch

        idx = self._design_idx(designs)
        outs = []
        with self._lock, torch.inference_mode():
            for s in range(0, len(designs), batch_size):
                oh = torch.nn.functional.one_hot(idx[s : s + batch_size], 4).float()
                outs.append(self.model(self._onehot_window(oh))["mouse"][:, :, tracks].permute(0, 2, 1))
        return {"raw": _pack(torch.cat(outs))}


@app.cls(
    image=borzoi_image,
    gpu="H100",
    max_containers=BORZOI_MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_S,
    timeout=900,
)
@modal.concurrent(max_inputs=8, target_inputs=3)
class BorzoiOracle(_Common):
    ctx, n_up, bin_bp = BOR_CTX, BOR_UP, BOR_BIN

    @modal.enter()
    def load(self) -> None:
        import threading

        import torch
        from borzoi_pytorch import Borzoi

        torch.backends.cudnn.benchmark = True
        self._lock = threading.Lock()
        self.models = []
        self.full_heads = []
        for r in range(BOR_REPLICATES):
            m = Borzoi.from_pretrained(f"johahi/borzoi-replicate-{r}-mouse")
            self.full_heads.append(m.mouse_head)
            head = torch.nn.Conv1d(1920, 1, 1)
            head.weight.data = m.mouse_head.weight.data[BOR_TRACK : BOR_TRACK + 1].clone()
            head.bias.data = m.mouse_head.bias.data[BOR_TRACK : BOR_TRACK + 1].clone()
            m.mouse_head = head
            m = m.cuda().eval()
            self.full_heads[-1] = self.full_heads[-1].cuda()
            for p in m.parameters():
                p.requires_grad_(False)
            self.models.append(m)
        self._init_flanks()

    def _raw(self, window_oh):
        """-> (B, 4 replicates, 6144) raw predictions, autocast exactly as the deployed predictor."""
        import torch

        x = window_oh.transpose(1, 2).contiguous()
        outs = []
        with torch.autocast(device_type="cuda"):
            for m in self.models:
                outs.append(m(x, is_human=False)[:, 0, :].float())
        return torch.stack(outs, dim=1)

    @staticmethod
    def _loss(raw, target):
        import torch

        p = raw / raw.amax(dim=(1, 2), keepdim=True)
        prof = p.mean(dim=1) - torch.std(p, dim=1, correction=0)
        return (target.unsqueeze(0) - prof[:, : target.numel()]).abs().sum(1), prof

    @modal.method()
    def score(self, designs: list[str], open_bp: int = 1664, closed_bp: int = 1664, return_raw: bool = False,
              batch_size: int = 2) -> dict:
        import torch

        idx = self._design_idx(designs)
        target = self._target_bins(idx.shape[1], BOR_BIN, open_bp, closed_bp)
        losses, raws = [], []
        with self._lock, torch.inference_mode():
            t0 = time.time()
            for s in range(0, len(designs), batch_size):
                oh = torch.nn.functional.one_hot(idx[s : s + batch_size], 4).float()
                raw = self._raw(self._onehot_window(oh))
                loss, _ = self._loss(raw, target)
                losses.append(loss)
                raws.append(raw)
            torch.cuda.synchronize()
        out = {"f": torch.cat(losses).tolist(), "gpu_s": time.time() - t0}
        if return_raw:
            out["raw"] = _pack(torch.cat(raws))
        return out

    @modal.method()
    def score_and_grad(self, designs: list[str], open_bp: int = 1664, closed_bp: int = 1664,
                       loss_scale: float = 64.0) -> dict:
        """Loss and d(loss)/d(design one-hot). Backward runs in fp16 autocast, so the loss is scaled to avoid underflow."""
        import torch

        idx = self._design_idx(designs)
        target = self._target_bins(idx.shape[1], BOR_BIN, open_bp, closed_bp)
        with self._lock:
            t0 = time.time()
            # fp16 backward can overflow at a large scale and underflow at a small one; back off until finite.
            for scale in (loss_scale, loss_scale / 16, loss_scale / 256, 1.0):
                x = torch.nn.functional.one_hot(idx, 4).float().requires_grad_(True)
                raw = self._raw(self._onehot_window(x))
                loss, _ = self._loss(raw, target)
                (loss.sum() * scale).backward()
                grad = x.grad / scale
                if bool(torch.isfinite(grad).all().item()):
                    break
            torch.cuda.synchronize()
        return {
            "f": loss.detach().tolist(),
            "grad": _pack(grad),
            "finite": bool(torch.isfinite(grad).all().item()),
            "scale": scale,
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_s": time.time() - t0,
        }

    @modal.method()
    def raw_tracks(self, designs: list[str], tracks: list[int], batch_size: int = 2) -> dict:
        """Raw mouse-head predictions of the 4 replicates for arbitrary tracks (held-out checks) -> (B, 4, T, 6144)."""
        import torch

        idx = self._design_idx(designs)
        outs = []
        with self._lock, torch.inference_mode():
            for s in range(0, len(designs), batch_size):
                oh = torch.nn.functional.one_hot(idx[s : s + batch_size], 4).float()
                x = self._onehot_window(oh).transpose(1, 2).contiguous()
                reps = []
                with torch.autocast(device_type="cuda"):
                    for m, full in zip(self.models, self.full_heads):
                        sliced, m.mouse_head = m.mouse_head, full
                        try:
                            reps.append(m(x, is_human=False)[:, tracks, :].float())
                        finally:
                            m.mouse_head = sliced
                outs.append(torch.stack(reps, dim=1))
        return {"raw": _pack(torch.cat(outs))}
