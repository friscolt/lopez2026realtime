# Inference demo

Runs the trained ResNet34 baseline (full image + ROI) and the ProtoNet (ROI) model on a handful of
real test images, and produces:

- A table + JSON of predictions (predicted class, confidence, correct/incorrect) for N test images per class.
- A Grad-CAM comparison figure (full-image model vs ROI model) for those same images.
- A UMAP plot of the full test set for all three models (ResNet34 full, ResNet34 ROI, ProtoNet ROI),
  with the chosen examples highlighted with a star marker.

## Requirements

Trained checkpoints already present under `$DATA_ROOT` (see main [README](../README.md)):

```
models/resnet34_baseline/resnet34_<view>_seed<seed>_best.pth
models/resnet34_roi_baseline/resnet34_roi_<view>_seed<seed>_best.pth
models_fsl/protonet_resnet34_roi/protonet_resnet34/<view>/<shot>shot/model.pth
```

## Usage

```bash
export DATA_ROOT=/mnt   # or wherever your models/data live
python demo/run_inference_demo.py --view MIX --seed 0 --shot 5 --n_per_class 1
```

Outputs are written to `demo/outputs/` (gitignored): `inference_results.json`, `gradcam_examples.png`,
`umap_highlighted.png`.

Note: UMAP is computed over the *entire* test set of the chosen view, not just the highlighted
examples — a handful of points isn't enough for UMAP's neighbor graph to produce a meaningful layout.
