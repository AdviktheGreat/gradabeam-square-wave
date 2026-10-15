#!/bin/sh
# GrAdaBeam from an order-5 Markov start (trained on mm39 autosomes), library-default settings, 13k fe, seeds 0-3.
# Seeds 0 and 1 pair with default_s0/default_s1 (same GrAdaBeam seed and hash seed, only the start sequence differs).
cd "$(dirname "$0")"
export PYTHONPATH=".:${GRADABEAM_PATH:?set GRADABEAM_PATH to a checkout of move37-labs/gradabeam}"
PY=${PY:-python}
mkdir -p results/logs
for s in 0 1 2 3; do
  PYTHONHASHSEED=$s $PY run_gradabeam.py --seed $s --budget-fe 13000 --name markov_default_s$s --start markov --beam-size 2 --rollouts 4 --mut-rate 2.0 --alpha 0.5 > results/logs/markov_default_s$s.log 2>&1 &
done
wait
