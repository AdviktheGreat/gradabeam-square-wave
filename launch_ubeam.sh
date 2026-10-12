#!/bin/sh
# Uniform-proposal beam search controls: 30 tok/bp (K'=2,K=15) x3 seeds and 84 tok/bp (K'=2,K=42) x2 seeds.
cd "$(dirname "$0")"
export PYTHONPATH=.
PY=${PY:-python}
mkdir -p results/logs
for s in 0 1 2; do
  $PY uniform_beam.py --kprime 2 --k 15 --seed $s > results/logs/ubeam_kp2_k15_s$s.log 2>&1 &
done
for s in 0 1; do
  $PY uniform_beam.py --kprime 2 --k 42 --seed $s > results/logs/ubeam_kp2_k42_s$s.log 2>&1 &
done
wait
