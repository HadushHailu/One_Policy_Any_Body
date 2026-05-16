#!/bin/bash
# E4: Morphology Feature Ablation — Figure 5-6
#
# Proves Claim 5: Morphology features are interpretable and minimal
#
# Removes one feature group at a time:
#   - no_dh: remove DH parameters
#   - no_limits: remove joint limits
#   - no_workspace: remove workspace volume
#   - no_payload: remove payload/gripper features
#   - random: replace morphology with random noise (sanity check)

set -euo pipefail

SEEDS="42 43"
ABLATIONS="full no_dh no_limits no_workspace no_payload random"

echo "=== E4: Morphology Feature Ablation ==="

for ablation in $ABLATIONS; do
    for seed in $SEEDS; do
        echo "--- Ablation: $ablation, seed=$seed ---"
        python train.py policy=morphology_dp \
            task=franka_pick,ur5_pick \
            robot=franka,ur5 \
            policy.unet.use_film=true \
            morphology_ablation=$ablation \
            training.seed=$seed \
            logging.group=E4_ablation_${ablation}
    done
done

echo "=== E4 complete ==="
