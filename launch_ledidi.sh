#!/bin/sh
# Ledidi from an order-5 Markov start (trained on mm39 autosomes), library-default settings (l=0.1, batch 16, lr 1.0), 13k fe, seeds 0-3.
# Seeds 0-3 use the same start sequences as markov_default_s0-3, so each Ledidi run pairs with a GrAdaBeam run.
cd "$(dirname "$0")"
export PYTHONPATH=.
PY=${PY:-python}
mkdir -p results/logs
for s in 0 1 2 3; do
  $PY run_ledidi.py --seed $s --budget-fe 13000 --l 0.1 --name ledidi_markov_s$s > results/logs/ledidi_markov_s$s.log 2>&1 &
done
wait
