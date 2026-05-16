#!/usr/bin/env bash
# Download pre-collected demonstration data (placeholder).
# Replace URLs with actual dataset hosting (HuggingFace, GDrive, etc.)
set -euo pipefail

DATA_DIR=${1:-"data"}
mkdir -p "${DATA_DIR}"

echo "===== Download Demonstration Data ====="
echo "Target directory: ${DATA_DIR}"
echo ""

# Placeholder — update URLs once data is hosted
DATASETS=(
  "franka_pick"
  "ur5_pick"
  "so101_pick"
)

for ds in "${DATASETS[@]}"; do
  if [ -d "${DATA_DIR}/${ds}" ]; then
    echo "[SKIP] ${ds} already exists"
  else
    echo "[TODO] ${ds} — URL not yet configured"
    # Example:
    # wget -O "${DATA_DIR}/${ds}.tar.gz" "https://huggingface.co/datasets/hadush/opab/${ds}.tar.gz"
    # tar -xzf "${DATA_DIR}/${ds}.tar.gz" -C "${DATA_DIR}"
    # rm "${DATA_DIR}/${ds}.tar.gz"
    mkdir -p "${DATA_DIR}/${ds}"
    echo "  Created placeholder directory ${DATA_DIR}/${ds}"
  fi
done

echo ""
echo "Done. Update this script with real URLs once data is hosted."
