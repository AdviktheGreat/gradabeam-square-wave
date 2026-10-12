#!/bin/sh
# Pilot: choose mutations_per_sequence on pilot seeds (100+) that are never reused for the main runs.
cd "$(dirname "$0")"
export PYTHONPATH=".:${GRADABEAM_PATH:?set GRADABEAM_PATH to a checkout of move37-labs/gradabeam}"
PY=${PY:-python}
mkdir -p results/logs
for rate in 2 20 200 1000; do
  PYTHONHASHSEED=100 $PY run_gradabeam.py --seed 100 --budget-fe 2000 --mut-rate $rate --name pilot_rate${rate} > results/logs/pilot_rate${rate}.log 2>&1 &
done
wait
