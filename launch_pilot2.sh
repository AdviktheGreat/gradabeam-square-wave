#!/bin/sh
# Pilot 2: beam / rollouts / exploration_alpha at mutations_per_sequence=200 on pilot seed 101 (never reused for main runs).
cd "$(dirname "$0")"
export PYTHONPATH=".:${GRADABEAM_PATH:?set GRADABEAM_PATH to a checkout of move37-labs/gradabeam}"
PY=${PY:-python}
mkdir -p results/logs
run() { # name beam rollouts rate alpha
  PYTHONHASHSEED=101 $PY run_gradabeam.py --seed 101 --budget-fe 3000 --name pilot2_$1 --beam-size $2 --rollouts $3 --mut-rate $4 --alpha $5 > results/logs/pilot2_$1.log 2>&1 &
}
run A_b2_r4_a0.5   2 4  200 0.5
run B_b2_r16_a0.5  2 16 200 0.5
run C_b8_r4_a0.5   8 4  200 0.5
run D_b8_r16_a0.5  8 16 200 0.5
run E_b2_r4_a0.1   2 4  200 0.1
run F_b2_r4_a0.9   2 4  200 0.9
run G_b2_r16_m50   2 16 50  0.5
run H_b4_r8_a0.1   4 8  200 0.1
wait
