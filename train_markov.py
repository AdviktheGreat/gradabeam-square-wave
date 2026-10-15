"""Train the order-5 Markov chain on the mm39 autosomes (chr1-19) and sanity-check what it generates.

    sh fetch_autosomes.sh                  # or download chr1..chr19 .fa.gz from UCSC into data/mm39_autosomes/
    PYTHONPATH=. python train_markov.py

chrX is left out on purpose so the designed locus is not in the training data. The chain has 4**5 * 3 free parameters, so it
cannot memorise any locus anyway.
"""

import time
from pathlib import Path

import numpy as np

from squarewave import scoring as S
from squarewave.markov import counts_to_probs, count_kmers, read_fasta_codes, sample
from squarewave.metrics import cpg_obs_exp, dinuc_freq

ORDER = 5
FASTA_DIR = S.DATA / "mm39_autosomes"
OUT = S.DATA / "markov_o5_mm39_autosomes.npz"


def main() -> None:
    chroms = [f"chr{i}" for i in range(1, 20)]
    counts = np.zeros((4**ORDER, 4), dtype=np.int64)
    for c in chroms:
        t = time.time()
        codes = read_fasta_codes(FASTA_DIR / f"{c}.fa.gz")
        counts += count_kmers(codes, ORDER)
        print(f"{c}: {len(codes):,} bp, {(codes < 4).mean():.3f} ACGT, total k-mers so far {counts.sum():,} ({time.time() - t:.0f}s)", flush=True)
    np.savez_compressed(OUT, counts=counts, order=ORDER, chroms=np.array(chroms))
    print("saved", OUT)

    probs = counts_to_probs(counts)
    base = counts.sum(axis=0) / counts.sum()
    print("base composition A,C,G,T:", np.round(base, 4), "GC %.4f" % (base[1] + base[2]))

    up, _ = S.load_flanks()
    native = (S.DATA / "mm39_chrX_native_region.txt").read_text().strip()
    ref = dinuc_freq(native)
    rng = np.random.default_rng(0)
    print("\nsampled 19,584 bp sequences vs chrX native region (GC %.3f, CpG o/e %.3f)" % ((native.count("G") + native.count("C")) / len(native), cpg_obs_exp(native)))
    for k in range(4):
        s = sample(probs, 19_584, up[-ORDER:], rng)
        print(f"  sample {k}: GC {(s.count('G') + s.count('C')) / len(s):.3f}  CpG o/e {cpg_obs_exp(s):.3f}  dinuc L1 vs native {np.abs(dinuc_freq(s) - ref).sum():.3f}")
    u = "".join(rng.choice(list("ACGT"), size=19_584))
    print(f"  uniform  : GC {(u.count('G') + u.count('C')) / len(u):.3f}  CpG o/e {cpg_obs_exp(u):.3f}  dinuc L1 vs native {np.abs(dinuc_freq(u) - ref).sum():.3f}")


if __name__ == "__main__":
    main()
