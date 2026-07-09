#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt}"

FULL_DATA="$DATA_ROOT"
ROI_DATA="$DATA_ROOT/roi_dataset_v4"

FULL_SAVE="$DATA_ROOT/models_fsl/protonet_resnet34_full"
ROI_SAVE="$DATA_ROOT/models_fsl/protonet_resnet34_roi"

VIEWS=("SUR" "SEC" "MIX")
SHOTS=(1 3 5)

echo "========================================"
echo "Running FEW-SHOT ProtoNet Experiments"
echo "========================================"

########################################
# FULL IMAGE EXPERIMENTS
########################################

echo ""
echo "========================================"
echo "FULL IMAGE EXPERIMENTS"
echo "========================================"

for VIEW in "${VIEWS[@]}"
do
  for SHOT in "${SHOTS[@]}"
  do

    echo ""
    echo "----------------------------------------"
    echo "Training FULL | View: $VIEW | Shot: $SHOT"
    echo "----------------------------------------"

    python $SCRIPT_DIR/train_protonet.py \
      --view $VIEW \
      --shot $SHOT \
      --data_root $FULL_DATA \
      --save_root $FULL_SAVE

    echo "Evaluating FULL | View: $VIEW | Shot: $SHOT"

    python $SCRIPT_DIR/evaluate_protonet.py \
      --view $VIEW \
      --shot $SHOT \
      --data_root $FULL_DATA \
      --model_root $FULL_SAVE

  done
done


########################################
# ROI EXPERIMENTS
########################################

echo ""
echo "========================================"
echo "ROI EXPERIMENTS"
echo "========================================"

for VIEW in "${VIEWS[@]}"
do
  for SHOT in "${SHOTS[@]}"
  do

    echo ""
    echo "----------------------------------------"
    echo "Training ROI | View: $VIEW | Shot: $SHOT"
    echo "----------------------------------------"

    python $SCRIPT_DIR/train_protonet.py \
      --view $VIEW \
      --shot $SHOT \
      --data_root $ROI_DATA \
      --save_root $ROI_SAVE

    echo "Evaluating ROI | View: $VIEW | Shot: $SHOT"

    python $SCRIPT_DIR/evaluate_protonet.py \
      --view $VIEW \
      --shot $SHOT \
      --data_root $ROI_DATA \
      --model_root $ROI_SAVE

  done
done


echo ""
echo "========================================"
echo "ALL FEW-SHOT EXPERIMENTS COMPLETED"
echo "========================================"
