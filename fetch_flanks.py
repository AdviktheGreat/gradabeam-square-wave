"""Rebuild (or verify) the mm39 chrX flank and native-region files in data/ from the UCSC mm39 chromosome FASTA.

    python fetch_flanks.py            # download chrX.fa.gz from UCSC, compare with data/*.txt
    python fetch_flanks.py --fasta chrX.fa.gz --write

Region: chrX:52,051,929-52,123,468 (1-based, inclusive; 71,540 bp). The upstream flank is the 163,840 bp before it,
the downstream flank is the 400,000 bp after it.
"""

import argparse
import gzip
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
URL = "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes/chrX.fa.gz"
START0, END0 = 52_051_928, 52_123_468  # 0-based, half-open
UP, DOWN = 163_840, 400_000


def read_chrx(path: Path) -> str:
    with gzip.open(path, "rt") as f:
        lines = f.read().split("\n")
    assert lines[0].startswith(">chrX"), lines[0]
    return "".join(lines[1:]).upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, default=None, help="local chrX.fa.gz; downloaded from UCSC if omitted")
    ap.add_argument("--write", action="store_true", help="overwrite data/*.txt instead of only comparing")
    a = ap.parse_args()

    fasta = a.fasta
    if fasta is None:
        fasta = DATA / "chrX.fa.gz"
        if not fasta.exists():
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, fasta)
    chrx = read_chrx(fasta)
    pieces = {
        "mm39_chrX_flank_up163840.txt": chrx[START0 - UP : START0],
        "mm39_chrX_native_region.txt": chrx[START0:END0],
        "mm39_chrX_flank_down400000.txt": chrx[END0 : END0 + DOWN],
    }
    for name, seq in pieces.items():
        path = DATA / name
        if a.write:
            path.write_text(seq)
            print(f"wrote {path} ({len(seq)} bp)")
        else:
            same = path.exists() and path.read_text().strip() == seq
            print(f"{name}: {len(seq)} bp, {'matches' if same else 'DIFFERS from'} the committed file")


if __name__ == "__main__":
    main()
