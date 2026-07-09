# Results

Trained checkpoints and per-run metrics (CSV/JSON) are **not stored in this repository** — they live
under `$DATA_ROOT/models/` and `$DATA_ROOT/models_fsl/` (see the main [README](../README.md)'s "Expected
data layout" section). This file only holds a static snapshot of the numbers reported in the paper /
original experiment run, for reference.

## Example results — MIX view, seed 0

From the original experiment run; exact numbers depend on your data split and are only reproducible
with the same train/test partition.

| Model | Accuracy | Weighted F1 |
|---|---|---|
| ResNet34, full image | 70.4% | 70.2% |
| ResNet34, full image + TTA | 70.4% | 70.0% |
| ResNet34, ROI | 98.3% | 98.3% |
| ResNet34, ROI + TTA | 98.3% | 98.3% |
| ProtoNet, ROI (5-shot, 1000 episodes) | 97.0% ± 3.0% | — |
| ProtoNet, ROI (5-shot) + TTA | 96.7% ± 3.4% | — |

ROI cropping gives a large accuracy jump over full-image classification on this view; the same pattern
holds for ResNet50 and ViT-small. See [demo/](../demo/) for a runnable inference + Grad-CAM + UMAP
comparison, and `scripts/summarize_results.py` / `scripts/summarize_fewshot_results.py` to aggregate
results across all views and seeds from your own runs.

## Best checkpoint per view (all backbones/shots compared)

Selected by comparing accuracy across all 3 training seeds (baseline) or all shot counts (ProtoNet):

**ResNet34 + ROI** (best backbone overall — matches or beats ResNet50/ViT-small with the lightest model):

| View | Seed | Accuracy |
|---|---|---|
| MIX | 0 | 98.3% |
| SEC | 0 | 98.3% |
| SUR | 0 | 95.0% |

**ProtoNet + ROI:**

| View | Shot | Accuracy |
|---|---|---|
| MIX | 5-shot | 97.0% ± 3.0% |
| SEC | 3-shot | 99.5% ± 1.6% (beats 5-shot: 96.6%) |
| SUR | 5-shot | 96.3% ± 3.0% |
