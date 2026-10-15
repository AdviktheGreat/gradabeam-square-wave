#!/bin/sh
# Download mm39 chr1..chr19 (about 750 MB) from UCSC into data/mm39_autosomes/ for train_markov.py. Resumable.
set -e
cd "$(dirname "$0")"
mkdir -p data/mm39_autosomes
cd data/mm39_autosomes
seq 1 19 | xargs -P 4 -I{} curl -sS -f -C - -O https://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes/chr{}.fa.gz
for f in chr*.fa.gz; do gzip -t "$f"; done
echo "downloaded and verified $(ls chr*.fa.gz | wc -l) chromosomes"
