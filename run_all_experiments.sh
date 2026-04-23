#!/bin/bash
# run_all_experiments.sh

echo "================================================="
echo "Batteria esperimenti MMD vs Overlap Area"
echo "Dataset: Wine, Cancer, Digits"
echo "================================================="

mkdir -p results/test_npoints_300
mkdir -p results/test_npoints_10000
mkdir -p results/test_nsamples_20000
mkdir -p results/test_nsamples_100000
mkdir -p results/test_components_2
mkdir -p results/test_components_3

# ── TEST 1: variazione n_points ───────────────────────
echo ""
echo "TEST 1 — n_points=300"
for DATASET in wine cancer digits; do
    echo "  >>> $DATASET"
    python3 experiments/mmd_overlap_analysis.py \
        --dataset      $DATASET \
        --n_shifts     100 \
        --n_configs    100 \
        --n_points     300 \
        --n_samples    20000 \
        --n_components 2 \
        --output_dir   results/test_npoints_300
done

echo ""
echo "TEST 1 — n_points=10000"
for DATASET in wine cancer digits; do
    echo "  >>> $DATASET"
    python3 experiments/mmd_overlap_analysis.py \
        --dataset      $DATASET \
        --n_shifts     100 \
        --n_configs    100 \
        --n_points     10000 \
        --n_samples    20000 \
        --n_components 2 \
        --output_dir   results/test_npoints_10000
done

# ── TEST 2: variazione n_samples ──────────────────────
echo ""
echo "TEST 2 — n_samples=20000"
for DATASET in wine cancer digits; do
    echo "  >>> $DATASET"
    python3 experiments/mmd_overlap_analysis.py \
        --dataset      $DATASET \
        --n_shifts     100 \
        --n_configs    100 \
        --n_points     1000 \
        --n_samples    20000 \
        --n_components 2 \
        --output_dir   results/test_nsamples_20000
done

echo ""
echo "TEST 2 — n_samples=100000"
for DATASET in wine cancer digits; do
    echo "  >>> $DATASET"
    python3 experiments/mmd_overlap_analysis.py \
        --dataset      $DATASET \
        --n_shifts     100 \
        --n_configs    100 \
        --n_points     1000 \
        --n_samples    100000 \
        --n_components 2 \
        --output_dir   results/test_nsamples_100000
done

# ── TEST 3: variazione n_components ───────────────────
echo ""
echo "TEST 3 — n_components=2"
for DATASET in wine cancer digits; do
    echo "  >>> $DATASET"
    python3 experiments/mmd_overlap_analysis.py \
        --dataset      $DATASET \
        --n_shifts     100 \
        --n_configs    100 \
        --n_points     1000 \
        --n_samples    20000 \
        --n_components 2 \
        --output_dir   results/test_components_2
done

echo ""
echo "TEST 3 — n_components=3"
for DATASET in wine cancer digits; do
    echo "  >>> $DATASET"
    python3 experiments/mmd_overlap_analysis.py \
        --dataset      $DATASET \
        --n_shifts     100 \
        --n_configs    100 \
        --n_points     1000 \
        --n_samples    20000 \
        --n_components 3 \
        --output_dir   results/test_components_3
done

echo ""
echo "================================================="
echo "Tutti i test completati."
echo "================================================="