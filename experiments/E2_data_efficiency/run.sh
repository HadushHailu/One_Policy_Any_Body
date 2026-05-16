#!/bin/bash
# E2: Data Efficiency — Figure 3
#
# Proves Claim 2: Morphology descriptors reduce data requirements
#
# Varies the number of SO-101 demos: 0, 5, 10, 25, 50, 100
# Compares morph-conditioned vs. baseline at each data level.

set -euo pipefail

SEEDS="42 43 44"
N_DEMOS="0 5 10 25 50 100"

echo "=== E2: Data Efficiency Curve ==="

for n in $N_DEMOS; do
    for seed in $SEEDS; do
        echo "--- Ours: n_demos=$n, seed=$seed ---"
        python train.py policy=morphology_dp \
            task=franka_pick,ur5_pick,so101_pick \
            robot=franka,ur5,so101 \
            policy.unet.use_film=true \
            training.seed=$seed \
            dataset.so101_max_demos=$n \
            logging.group=E2_data_efficiency_n${n}
    done
done

echo "=== E2 complete. Run experiments/E2_data_efficiency/plot.py ==="
