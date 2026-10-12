#!/bin/sh
# Main runs, 13k forward-equivalents each (Evo 2 long wave: 153 chunks x 84 tok/bp ~ 12.9k scored candidates).
# tuned = pilot-2 config H (beam 4, rollouts 8, mutations_per_sequence 200, alpha 0.1); default = the README example setting.
cd "$(dirname "$0")"
export PYTHONPATH=".:${GRADABEAM_PATH:?set GRADABEAM_PATH to a checkout of move37-labs/gradabeam}"
PY=${PY:-python}
mkdir -p results/logs
for s in 0 1 2 3; do
  PYTHONHASHSEED=$s $PY run_gradabeam.py --seed $s --budget-fe 13000 --name main_s$s --beam-size 4 --rollouts 8 --mut-rate 200 --alpha 0.1 > results/logs/main_s$s.log 2>&1 &
done
for s in 0 1; do
  PYTHONHASHSEED=$s $PY run_gradabeam.py --seed $s --budget-fe 13000 --name default_s$s --beam-size 2 --rollouts 4 --mut-rate 2.0 --alpha 0.5 > results/logs/default_s$s.log 2>&1 &
done
wait
