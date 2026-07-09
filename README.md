# Kidney Stone Subtype Classification — Baselines, ROI, Few-Shot & Grad-CAM

![CI](https://github.com/friscolt/lopez2026realtime/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Research code for classifying kidney stone (Randall's plug / calculi) subtypes from endoscopic images
across three acquisition views — **SUR** (surface), **SEC** (section), **MIX** (combined) — using
CNN/ViT baselines, region-of-interest (ROI) cropping, test-time augmentation (TTA), few-shot learning
(ProtoNet), Grad-CAM interpretability, and inference-time benchmarks.

## Pipeline overview

```
                 ┌────────────────────┐
                 │  raw images + masks │  $DATA_ROOT/{SUR,SEC,MIX}/{train,test}/<class>/
                 └─────────┬──────────┘
                           │ baseline/scripts/create_roi_dataset_v4.py
                           ▼
                 ┌────────────────────┐
                 │   ROI dataset v4    │  $DATA_ROOT/roi_dataset_v4/...
                 └─────────┬──────────┘
             ┌─────────────┼─────────────────────────┐
             ▼                                        ▼
   ┌───────────────────┐                    ┌───────────────────────┐
   │ baseline train/eval│                    │ few-shot ProtoNet      │
   │ (ResNet34/50, ViT-S)│                    │ train/eval/eval+TTA   │
   │ full image & ROI,   │                    │ (fewshot/scripts/)    │
   │ +TTA, +summarize    │                    └───────────┬───────────┘
   │ (baseline/scripts/) │                                │
   └─────────┬───────────┘                                │
             │                                             ▼
             ▼                                   plot_tsne / plot_umap
   gradcam/, plot_gradcam_*.py                 (embedding comparison)
   (interpretability)

   time/  →  inference latency benchmarks (full image vs ROI, baseline vs few-shot)
```

## Repository structure

```
baseline/scripts/     ResNet34/50 & ViT-small: train, evaluate (+TTA), summarize, ROI dataset creation
baseline/scripts/legacy/  superseded scripts (older/unversioned ROI dataset, one-off preprocessing)
fewshot/scripts/       ProtoNet: episodic dataset, model, train/evaluate(+TTA), embedding plots
gradcam/               Grad-CAM generation for the ResNet34 baseline (full & ROI)
time/                  Inference latency benchmarks
```

## Installation

Requires **Python 3.10+**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions this codebase was verified against (`torch==2.7.1`,
`torchvision==0.22.1`, built for CUDA 11.8). If you need a different CUDA version, install the matching
`torch`/`torchvision` build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then
`pip install -r requirements.txt` for the rest.

## Expected data layout

All scripts read/write under a single root, configurable via the `DATA_ROOT` environment variable
(defaults to `/mnt` if unset — matching the layout this project was originally developed on):

```
$DATA_ROOT/
├── SUR/{train,test}/<class>/*.png            (+ <class>/mask/*.png for ROI extraction)
├── SEC/{train,test}/<class>/*.png            (+ mask/)
├── MIX/{train,test}/<class>/*.png            (+ mask/)
├── roi_dataset_v4/{SUR,SEC,MIX}/{train,test}/<class>/*.png   (generated)
├── models/<backbone>_baseline/                (baseline checkpoints + metrics)
├── models_fsl/<experiment>/<view>/<k>shot/    (ProtoNet checkpoints + eval results)
└── gradcams/, figures/                        (Grad-CAM / plotting outputs)
```

Each class folder is expected in `torchvision.datasets.ImageFolder` layout. For ROI extraction, place a
binary mask per image under a `mask/` subfolder next to it (same basename, `.png`/`.jpg`/`.jpeg`).

```bash
export DATA_ROOT=/path/to/your/data   # otherwise defaults to /mnt
```

### Data availability

The underlying endoscopic kidney stone images are clinical data and are **not distributed in this
repository**. To request access, contact the authors (see [CITATION.cff](CITATION.cff)). Everything
under `$DATA_ROOT` in the layout above is expected to come from that dataset (or your own, in the same
`ImageFolder` structure) — the code itself does not depend on any specific data source.

## Usage

### 1. Build the ROI dataset (optional, only needed for ROI experiments)

```bash
python baseline/scripts/create_roi_dataset_v4.py
```

### 2. Train & evaluate baselines

```bash
# Full-image baseline (ResNet34, view MIX, seed 0)
python baseline/scripts/train_resnet34_baseline.py --view MIX --seed 0
python baseline/scripts/evaluate_resnet34_tta.py --view MIX --seed 0

# ROI variant
python baseline/scripts/train_resnet34_baseline_roi.py --view MIX --seed 0
python baseline/scripts/evaluate_resnet34_roi_tta.py --view MIX --seed 0
```

The same pattern applies to `resnet50` and `vit_small`. After running all views/seeds, aggregate results with
the matching `summarize_*.py` script (e.g. `summarize_baseline_results_resnet34.py`).

### 3. Few-shot learning (ProtoNet)

```bash
cd fewshot/scripts
python train_protonet.py --view MIX --shot 5 --data_root "$DATA_ROOT"
python evaluate_protonet.py --view MIX --shot 5 --data_root "$DATA_ROOT"
python evaluate_protonet_tta.py --view MIX --shot 5 --data_root "$DATA_ROOT" --model_root "$DATA_ROOT/models_fsl"

# Or run the full sweep (all views × shots, full-image and ROI):
./run_fewshot_experiments.sh
./run_eval_all.sh
./run_eval_tta_roi.sh
```

`plot_tsne_compare.py` / `plot_umap_compare_clean.py` compare embedding spaces (ResNet34 full vs ROI vs
ProtoNet); they expect specific checkpoints (MIX/seed0, 5-shot) to already exist — edit the path constants
at the top to match your own runs.

### 4. Grad-CAM

```bash
python gradcam/generate_gradcam_resnet34_mix_seed0.py
python gradcam/generate_gradcam_resnet34_mix_seed0_roi.py
```

The `baseline/scripts/plot_gradcam_*.py` scripts build comparison figures from the generated overlays.

### 5. Demo: inference + Grad-CAM + UMAP on a few test examples

Once you have trained (or downloaded) checkpoints for the ResNet34 baseline (full + ROI) and the
ProtoNet (ROI) model, `demo/run_inference_demo.py` runs inference on a handful of real test images
(N per class), shows predictions with confidence, generates a Grad-CAM comparison figure, and plots
a UMAP of the full test set with those examples highlighted. See [demo/README.md](demo/README.md).

```bash
python demo/run_inference_demo.py --view MIX --seed 0 --shot 5 --n_per_class 1
```

### 6. Inference-time benchmarks

```bash
python time/test_time_resnet34_full_images.py   # ResNet34 baseline, full images
python time/test_time_ROI.py                    # ResNet34 baseline, ROI
python time/test_time_FSL.py                    # ProtoNet, full images
python time/test_time_resnet34_full_images_fewshot.py   # ProtoNet, full images (alt. loader)
```

All four expect the relevant checkpoints to already exist at `$DATA_ROOT/models/...` /
`$DATA_ROOT/models_fsl/...`.

## Results

Example results on the **MIX** view, seed 0 (from the original experiment run; exact numbers depend on
your data split and are only reproducible with the same train/test partition):

| Model | Accuracy | Weighted F1 |
|---|---|---|
| ResNet34, full image | 70.4% | 70.2% |
| ResNet34, full image + TTA | 70.4% | 70.0% |
| ResNet34, ROI | 98.3% | 98.3% |
| ResNet34, ROI + TTA | 98.3% | 98.3% |
| ProtoNet, ROI (5-shot, 1000 episodes) | 97.0% ± 3.0% | — |
| ProtoNet, ROI (5-shot) + TTA | 96.7% ± 3.4% | — |

ROI cropping gives a large accuracy jump over full-image classification on this view; see
[demo/](demo/) for a runnable inference + Grad-CAM + UMAP comparison, and `baseline/scripts/summarize_*.py`
/ `fewshot/scripts/eval_results_fsl.py` to aggregate results across all views and seeds.

## `legacy/`

`baseline/scripts/legacy/` (gitignored — not tracked in this repository, since it isn't part of the
reproducible pipeline) holds scripts kept only for local reference: ones that operate on superseded
dataset versions (`roi_dataset`/`roi_dataset_v3` instead of the current `roi_dataset_v4`), one-off
preprocessing utilities (`convert_images_to_png.py`), and superseded drafts (`plot_gradcam_full_mix.py`
loads the ROI checkpoint despite its name — fixed in `plot_gradcam_baseline_mix.py`).

## Roadmap

Continual-learning experiments (class-incremental training over the kidney stone subtypes, comparing
naive fine-tuning, experience replay, weight averaging, and LWF) exist as trained checkpoints on the
original development machine but are not yet implemented/published in this repository — they are
planned as a separate project.

## How to cite

If you use this code, please cite:

```bibtex
@inproceedings{lopez2026real,
  title={Real-time image segmentation for kidney stone identification using light-weight AI models},
  author={Lopez-Tiro, Francisco and Larose, Cl{\'e}ment and Doerler, Samuel and Ochoa-Ruiz, Gilberto and Daul, Christian},
  booktitle={Real-time Processing of Image, Depth and Video Information 2026},
  pages={18},
  year={2026},
  organization={SPIE}
}
```

See also [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).

## Author

F. Lopez-Tiro — Tecnologico de Monterrey & Université de Lorraine.
