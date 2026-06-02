#!/bin/bash
# Rerun all layout experiments after normalization fix + MC sampling fix
# Order: retrain → eval → missing-sensor eval
set -e

cd ~/deepOnet_solver/experiment

echo "================================================================"
echo "FULL RERUN: normalization fix + MC sampling fix"
echo "Started at $(date)"
echo "================================================================"

# Phase 1: Retrain Real-K12 (3 seeds)
echo ""
echo "=== Phase 1: Training Real-K12 ==="
for seed in 42 123 777; do
    echo "--- Real-K12 seed=$seed ---"
    python3 train_layout.py --layout sensors_real_stations.json --name real_k12_s${seed} --seed $seed
done

# Phase 2: Retrain Equispaced-K12 (3 seeds)
echo ""
echo "=== Phase 2: Training Equispaced-K12 ==="
for seed in 42 123 777; do
    echo "--- Equispaced-K12 seed=$seed ---"
    python3 train_layout.py --layout sensors_equispaced.json --name eq_k12_s${seed} --seed $seed
done

# Phase 3: Retrain Random-K12 (5 layout seeds, first 5 from available files)
echo ""
echo "=== Phase 3: Training Random-K12 (5 layouts) ==="
for lseed in 0 1 2 3 4; do
    echo "--- Random-K12 layout_seed=$lseed ---"
    python3 train_layout.py --layout sensors_random_seed${lseed}.json --name random_k12_ls${lseed} --seed 42
done

echo ""
echo "================================================================"
echo "ALL TRAINING DONE at $(date)"
echo "================================================================"

# Phase 4: Evaluate all checkpoints
echo ""
echo "=== Phase 4: Evaluation ==="
for seed in 42 123 777; do
    python3 eval_layout.py --layout sensors_real_stations.json --ckpt ckpt_real_k12_s${seed}.pt --name real_k12_s${seed}
done
for seed in 42 123 777; do
    python3 eval_layout.py --layout sensors_equispaced.json --ckpt ckpt_eq_k12_s${seed}.pt --name eq_k12_s${seed}
done
for lseed in 0 1 2 3 4; do
    python3 eval_layout.py --layout sensors_random_seed${lseed}.json --ckpt ckpt_random_k12_ls${lseed}.pt --name random_k12_ls${lseed}
done

# Phase 5: Missing-sensor eval (seed 42 only)
echo ""
echo "=== Phase 5: Missing-sensor robustness ==="
python3 eval_missing_sensors.py --layout sensors_real_stations.json --ckpt ckpt_real_k12_s42.pt --name real_k12_missing
python3 eval_missing_sensors.py --layout sensors_equispaced.json --ckpt ckpt_eq_k12_s42.pt --name eq_k12_missing

echo ""
echo "================================================================"
echo "ALL DONE at $(date)"
echo "================================================================"
