#!/bin/bash
set -euo pipefail

BASE="/root/shared-nvme/SD-AiO"
cd "${BASE}"

SCRIPTS=(
    # === Round 1: Random-shuffle baseline ===
    "scripts/train_restoration.sh"

    # === Round 2: Round-robin ablation ===
    "scripts/train_roundrobin.sh"
    "scripts/ablation_roundrobin/train_t50.sh"
    "scripts/ablation_roundrobin/train_t150.sh"
    "scripts/ablation_roundrobin/train_t200.sh"
    "scripts/ablation_roundrobin/train_t250.sh"
    "scripts/ablation_roundrobin/train_noVaeLora.sh"
    "scripts/ablation_roundrobin/train_resnet18.sh"
    "scripts/ablation_roundrobin/train_convnext.sh"
    "scripts/ablation_roundrobin/train_unetLora.sh"
)

echo "============================================"
echo "  SD-AiO Full Pipeline — $(date)"
echo "  Total experiments: ${#SCRIPTS[@]}"
echo "============================================"

for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    name=$(basename "${script%.sh}")

    echo ""
    echo "============================================"
    echo "[$(date)]  [$((i+1))/${#SCRIPTS[@]}]  Running: ${name}"
    echo "============================================"

    pkill -f 'train\.py' 2>/dev/null || true
    sleep 3

    bash "${script}"
    ret=$?

    if [ $ret -ne 0 ]; then
        echo "[$(date)]  FAILED: ${name} (exit=$ret)"
        exit $ret
    fi

    echo "[$(date)]  DONE: ${name}"
    sleep 5
done

echo ""
echo "============================================"
echo "  All experiments complete — $(date)"
echo "============================================"
