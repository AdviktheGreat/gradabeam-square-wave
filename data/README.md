# data/

| File | Source | In repo |
|---|---|---|
| `mm39_chrX_flank_up163840.txt`, `mm39_chrX_native_region.txt`, `mm39_chrX_flank_down400000.txt` | UCSC mm39 chrX, region chrX:52,051,929-52,123,468 plus flanks. Rebuild or verify with `python fetch_flanks.py`. | yes |
| `enformer_targets_mouse.txt`, `borzoi_targets_mouse.txt` | Track descriptions of the Enformer and Borzoi mouse heads (used to pick DNASE:129 ES-E14 and held-out ES tracks). | yes |
| `chrX.fa.gz` | UCSC mm39 chrX FASTA (52 MB). Downloaded by `fetch_flanks.py`. | no (gitignored) |
| `published_dnase_*.txt` | Evo 2 designed sequences, Supplementary Data 3 of Brixi et al., Nature 2026. | **no** |

## Evo 2 published sequences

The Evo 2 paper is licensed CC BY-NC-ND 4.0, so this repo does not redistribute its sequences. Only the long square wave
is needed to reproduce the baseline rows and `results/summary.png`:

1. Download Supplementary Data 3 from the paper's Nature page.
2. Save the long square wave design (19,584 bp, one line, ACGT only) as
   `data/published_dnase_squarewave_long_20kb.txt`.

The scripts that need it (`baselines.py`, `phase1_fetch.py`, `phase2_validate.py`, `make_figure.py`) raise a clear error
if the file is missing. Everything about the GrAdaBeam and uniform-beam runs is stored under `results/` without it.
