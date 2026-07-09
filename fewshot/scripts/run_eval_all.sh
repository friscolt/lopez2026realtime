#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt}"

FULL_DATA="$DATA_ROOT"
ROI_DATA="$DATA_ROOT/roi_dataset_v4"

FULL_MODELS="$DATA_ROOT/models_fsl/protonet_resnet34_full"
ROI_MODELS="$DATA_ROOT/models_fsl/protonet_resnet34_roi"

VIEWS=("SUR" "SEC" "MIX")
SHOTS=(1 3 5)

echo "============================================"
echo "RUNNING FEW-SHOT EVALUATIONS"
echo "============================================"

############################################
# FULL IMAGE MODELS
############################################

echo ""
echo "========== FULL IMAGE MODELS =========="

for VIEW in "${VIEWS[@]}"
do
  for SHOT in "${SHOTS[@]}"
  do

    echo ""
    echo "Evaluating FULL | View: $VIEW | Shot: $SHOT"

    python $SCRIPT_DIR/evaluate_protonet.py \
      --view $VIEW \
      --shot $SHOT \
      --data_root $FULL_DATA \
      --model_root $FULL_MODELS

  done
done


############################################
# ROI MODELS
############################################

echo ""
echo "========== ROI MODELS =========="

for VIEW in "${VIEWS[@]}"
do
  for SHOT in "${SHOTS[@]}"
  do

    echo ""
    echo "Evaluating ROI | View: $VIEW | Shot: $SHOT"

    python $SCRIPT_DIR/evaluate_protonet.py \
      --view $VIEW \
      --shot $SHOT \
      --data_root $ROI_DATA \
      --model_root $ROI_MODELS

  done
done


echo ""
echo "============================================"
echo "ALL EVALUATIONS FINISHED"
echo "============================================"