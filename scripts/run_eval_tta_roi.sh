#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/evaluate_protonet.py"
DATA_ROOT="${DATA_ROOT:-/mnt}"

DATA="$DATA_ROOT/roi_dataset_v4"
MODEL="$DATA_ROOT/models_fsl/protonet_resnet34_roi"

VIEWS=("SUR" "SEC" "MIX")
SHOTS=(1 3 5)

for SHOT in "${SHOTS[@]}"
do
  for VIEW in "${VIEWS[@]}"
  do
    echo "--------------------------------------"
    echo "Evaluating ROI TTA | Shot: $SHOT | View: $VIEW"
    echo "--------------------------------------"

    python $SCRIPT \
      --view $VIEW \
      --shot $SHOT \
      --data_root $DATA \
      --model_root $MODEL \
      --tta 5
  done
done
