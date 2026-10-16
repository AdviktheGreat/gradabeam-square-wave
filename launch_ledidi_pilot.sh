#!/bin/sh
# Pilot for the edit-penalty weight l: Ledidi from a Markov start, library-default batch size and lr, 5k fe, on a seed (100)
# that the main runs do not use. The paper loss (~100-200) is far larger than Ledidi's usual MSE scale, so its default l=0.1
# is not guaranteed to suit it. At the seed-0 start an edit is favoured when its first-order loss drop exceeds l, which is
# true for ~1.5% of positions at l=0.1, ~21% at l=0.01 and ~61% at l=0.001.
cd "$(dirname "$0")"
export PYTHONPATH=.
PY=${PY:-python}
mkdir -p results/logs
for l in 0.1 0.01 0.001; do
  $PY run_ledidi.py --seed 100 --budget-fe 5000 --l $l --name ledidi_pilot_l$l > results/logs/ledidi_pilot_l$l.log 2>&1 &
done
wait
