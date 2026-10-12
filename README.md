# GrAdaBeam on the Evo 2 long-square-wave design task

An independent test of [GrAdaBeam](https://github.com/move37-labs/gradabeam) on the chromatin-accessibility design task from
Evo 2 (Brixi et al., *Nature* 2026, Fig. 6 and SI A.6): design a ~19.6 kb mouse DNA sequence whose predicted DNase
accessibility follows a "long square wave" (1,664 bp open, 1,664 bp closed, open first), scored by the same oracle the Evo 2
paper uses (Enformer and Borzoi).

This is a small, independent reproduction. It is not affiliated with the GrAdaBeam or Evo 2 authors, and nothing here was tested in a wet lab.

## Result in one paragraph

With about the same number of oracle evaluations as Evo 2's published long wave (~13k scored candidates), GrAdaBeam started from
**uniform-random DNA** reaches loss 134 to 145 and AUROC 0.79 to 0.93 (6 runs). The published Evo 2 design scores loss 135.7 and
AUROC 0.998 under the same oracle. GrAdaBeam matches Evo 2 on the loss in two of six runs but not on AUROC, and it is roughly level
with a plain uniform-proposal beam search that has no gradients and no learned prior. The oracle budget is matched. Everything else
Evo 2 had is not (see [Is the comparison fair?](#is-the-comparison-fair)), so this does not show that GrAdaBeam is generally worse than Evo 2.

## Results

Final designs after ~13k forward-equivalents (fe). Loss is lower-is-better. AUROC is the mean of Enformer and Borzoi, as in the paper.
"Held-out" AUROC uses other mouse ES DNase tracks that the optimiser never saw (Enformer 10, 44, 45, 46; Borzoi 740, 774, 775, 776).
ICC is the agreement between Enformer and the four Borzoi replicates.

| Design | Runs | Loss | AUROC | Held-out AUROC | ICC |
|---|---|---|---|---|---|
| Evo 2 published long wave (rescored here) | 1 | 135.7 | 0.998 | 0.997 | 0.956 |
| GrAdaBeam, library defaults (beam 2, rollouts 4, rate 2.0, alpha 0.5) | 2 | 134.6, 134.1 | 0.926, 0.886 | 0.912, 0.891 | 0.96 to 0.97 |
| GrAdaBeam, pilot-tuned (beam 4, rollouts 8, rate 200, alpha 0.1) | 4 | 138 to 145 | 0.786 to 0.899 (mean 0.849) | 0.776 to 0.895 | 0.97 to 0.98 |
| Uniform-proposal chunked beam search, 84 tok/bp (K'=2, K=42) | 2 | 142.0, 152.0 | 0.873, 0.780 | 0.900, 0.784 | 0.97 to 0.98 |
| Uniform-proposal chunked beam search, 30 tok/bp (K'=2, K=15) | 3 | 163.6 to 177.6 | 0.849, 0.683, 0.880 | 0.855, 0.707, 0.872 | 0.83 to 0.95 |
| Uniform random / native mm39 region (references) | 1 each | 195.8 / 195.1 | 0.510 / 0.488 | 0.499 / 0.478 | 0.78 / 0.90 |

AUROC of the best-so-far design at intermediate budgets (6 GrAdaBeam runs): 0.58 to 0.70 at 500 fe, 0.71 to 0.82 at 4,000 fe,
0.73 to 0.87 at 6,000 fe, 0.79 to 0.93 at 13,000 fe. Loss was still decreasing at 13k fe, so these runs are not converged.
Evo 2's curve in the paper is about 0.955 at 30 tok/bp (~4.6k scored candidates), read by eye from Extended Data Fig. 10c (about ±0.02).

![Summary](results/summary.png)

*Top left: Evo 2 published design. Middle left: GrAdaBeam (default, seed 0). Bottom left: uniform-proposal beam search.
Top right: best loss so far against budget. Bottom right: AUROC against budget, with Evo 2's and the paper's uniform-proposal
curves read from the paper's figure.*

![Checks](results/checks.png)

*Left: GC fraction in open and closed segments. Only the Evo 2 design follows the wave in GC content. Right: seed-averaged AUROC
(optimised tracks and held-out tracks) and ICC.*

### What the designs look like

| | Evo 2 published | GrAdaBeam (6 runs) |
|---|---|---|
| GC, open / closed segments | 0.49 / 0.37 | 0.50 / 0.50 |
| CpG observed/expected | 0.64 | 0.99 to 1.04 |
| Dinucleotide statistics | genome-like | uniform-random-like |
| Edit distance from the random start | n/a | 19% to 63% |

GrAdaBeam edited a random sequence until the oracles saw a wave. The result does not have the sequence composition of natural
DNA. Held-out-track AUROC stays close to the optimised-track AUROC, so the designs are not just overfitting the two optimised
tracks, but the held-out tracks are other ES DNase experiments and are strongly correlated with the optimised ones.

## Setup

- **Region.** mm39 chrX:52,051,929-52,123,468. The native upstream and downstream flanks are fixed. Design length 19,584 bp, the
  length of the published long wave in Supplementary Data 3.
- **Oracle.** Enformer (mouse head, DNASE:129 ES-E14, track 11) and Borzoi mouse replicates 0 to 3 (track 741, mean minus std over
  replicates), max-normalised, L1 loss on the design bins, averaged: `f = (f_enformer + f_borzoi) / 2`.
- **Oracle validation.** The published Evo 2 long wave scores loss 135.71 (the proto-tools predictors give 135.70), AUROC 0.998, ICC
  0.956 (the paper reports about 0.95). Uniform random DNA scores 195.84.
- **Gradients.** GrAdaBeam needs `tism_torch`. The custom Modal oracle returns the loss and its gradient with respect to the one-hot
  input. Against real single-base substitutions the gradient has correlation 0.63 and sign agreement 0.83. Borzoi runs in fp16
  autocast, so the gradient uses a loss-scale backoff (default 64). A zero gradient is used only as a last resort and is counted
  (`n_bad_grad`, zero in the reported runs).
- **Budget unit.** Forward-equivalents. One scored candidate is 1 fe and one gradient is 2.1 fe, the measured H100 time ratio. Evo 2's
  long wave is 153 chunks x K'=2 x K=42 = 12,852 scored candidates. The GrAdaBeam runs use 13,000 fe, which is about 9.2k to 10.7k
  scored candidates plus 1.1k to 1.8k gradients.
- **Start.** Uniform-random ACGT per seed. The start sequences were not made easier.
- **GrAdaBeam.** The unmodified [`move37-labs/gradabeam`](https://github.com/move37-labs/gradabeam) at commit `1115891`,
  driven from Python. `PYTHONHASHSEED` is set to the seed because GrAdaBeam breaks ties through set iteration.
- **Tuning.** Two short pilots on seeds 100 and 101 (never reused for the main runs): mutations per sequence at 2k fe, then 8
  configurations at 3k fe. The tuned configuration did worse than the defaults at 13k fe (mean AUROC 0.85 against 0.91, but 4 and 2
  seeds, so this is not a significant difference).
- **Control.** A re-implementation of the paper's chunked beam search with a uniform-over-ACGT proposal in place of Evo 2
  (`uniform_beam.py`). At 30 tok/bp it gives AUROC 0.849, 0.683 and 0.880 (mean 0.804). The paper reports about 0.835 for its uniform
  generator at that budget, so this pipeline reproduces the paper's search behaviour. Partial designs are scored with the native
  sequence as the fill, which is an assumption about the paper's masking.

## Is the comparison fair?

The oracle budget is matched. Evo 2 also had advantages that GrAdaBeam did not:

- A pretrained 7B DNA model as the proposal, which is a strong prior. GrAdaBeam started from noise.
- A native 40,960 bp prompt of genomic context.
- About 1.6M generated tokens of generator compute on top of the oracle calls. This repo does not count generator compute.
- Settings tuned for this task, and a published design that was probably selected from several attempts.

GrAdaBeam had oracle gradients (charged at 2.1 fe) and only a small amount of tuning.

The most defensible reading: GrAdaBeam is about level with a no-prior uniform beam search under this oracle and budget, and the gap
to Evo 2's AUROC is likely mostly Evo 2's prior. With 2 to 6 seeds per arm and a wide spread between seeds (AUROC 0.79 to 0.90
within the tuned arm), small differences between GrAdaBeam variants should not be read as real.

## Limits

- The Borzoi mouse replicates here are the standard models, not Flashzoi as in the paper.
- The Evo 2 baseline numbers are rescores of the published sequence or values read off a figure.
- Only the long square wave was tested, with one oracle pairing and 13k fe. It was not run with more budget, so the claim is limited
  to this budget.
- Evo 2's own run was not reproduced. No wet-lab data.
- Modal cost for the whole project was roughly $50 (an estimate from the GPU-time ledger, not an invoice).

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/move37-labs/gradabeam && git -C gradabeam checkout 1115891
pip install -e gradabeam          # or just put it on PYTHONPATH
export GRADABEAM_PATH=$PWD/gradabeam

python fetch_flanks.py            # downloads UCSC mm39 chrX and checks data/*.txt (optional, files are committed)
modal setup                       # once
modal deploy squarewave/oracle_server.py [--env <env>]
export MODAL_ENVIRONMENT=<env>    # same env as the deploy; leave unset for the default

# one run: 13k fe, default GrAdaBeam settings
PYTHONHASHSEED=0 PYTHONPATH=.:$GRADABEAM_PATH python run_gradabeam.py --seed 0 --budget-fe 13000 --name my_run --beam-size 2 --rollouts 4 --mut-rate 2.0 --alpha 0.5
./launch_main.sh                  # all main runs (GRADABEAM_PATH must be set); also launch_ubeam.sh, launch_pilot*.sh
PYTHONPATH=. python analyze_runs.py main_s0 main_s1 --fe 500 1000 2000 4000 8000 13000
modal app stop gradabeam-squarewave ${MODAL_ENVIRONMENT:+-e $MODAL_ENVIRONMENT}   # when done
```

Each 13k-fe run took about 98 minutes of wall time and ~$3 of GPU time by the ledger. Baselines and `make_figure.py` need the Evo 2
published long wave, which is not redistributed here (see [`data/README.md`](data/README.md)).

## Layout

- `squarewave/oracle_server.py`: Modal app `gradabeam-squarewave` (Enformer fp32 service and Borzoi 4-replicate service; `score`, `score_and_grad`, `raw_tracks`).
- `squarewave/remote.py`, `gb_oracle.py`: threaded client with retry and a GPU-time ledger; GrAdaBeam oracle (`get_tism` is first-order loss differences).
- `squarewave/scoring.py`, `metrics.py`, `evaluate.py`: the paper's metrics (loss, AUROC, ICC), held-out-track AUROC and sequence statistics, in plain numpy.
- `run_gradabeam.py`, `uniform_beam.py`, `launch_*.sh`: GrAdaBeam runs and the uniform-proposal control.
- `analyze_runs.py`, `baselines.py`, `eval_ubeam.py`, `make_figure.py`: analysis and figures.
- `phase1_*.py`, `phase2_*.py`: oracle validation (proto-tools check, latency, gradient check).
- `results/`: per-run traces, milestone designs and best sequences (`runs/`), control runs (`uniform_beam/`), analysis JSON, figures, logs.

## Credits and licenses

Code in this repo is MIT-licensed ([LICENSE](LICENSE)). GrAdaBeam is Apache-2.0 and is used unmodified. Enformer, Borzoi and Evo 2 belong to
their authors. The Evo 2 paper is CC BY-NC-ND 4.0, which is why its sequences are not included here. Genome sequence is from UCSC mm39.

- Brixi et al., *Genome modelling and design across all domains of life with Evo 2*, Nature 2026.
- Avsec et al., Enformer, *Nature Methods* 2021. Linder et al., Borzoi, *Nature Genetics* 2025.
- GrAdaBeam preprint and code: [move37-labs/gradabeam](https://github.com/move37-labs/gradabeam).
