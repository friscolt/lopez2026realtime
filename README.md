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
                           │ scripts/build_roi_dataset.py
                           ▼
                 ┌────────────────────┐
                 │   ROI dataset v4    │  $DATA_ROOT/roi_dataset_v4/...
                 └─────────┬──────────┘
             ┌─────────────┼─────────────────────────┐
             ▼                                        ▼
   ┌───────────────────┐                    ┌───────────────────────┐
   │ scripts/train.py    │                    │ scripts/train_protonet.py │
   │ scripts/evaluate.py │                    │ scripts/evaluate_protonet.py│
   │ (ResNet34/50, ViT-S,│                    │ (few-shot, full & ROI) │
   │  full image & ROI)  │                    └───────────┬───────────┘
   └─────────┬───────────┘                                │
             │                                             ▼
             ▼                                   scripts/plot_embeddings.py
   scripts/generate_gradcam.py                  (t-SNE / UMAP comparison)
   scripts/plot_gradcam_*.py

   scripts/benchmark_inference_time.py  →  latency (full image vs ROI, classifier vs few-shot)
```

## Repository structure

```
config/
├── baseline/       hyperparameters per backbone x {full, roi} (resnet34, resnet50, vit_small)
├── fewshot/        ProtoNet hyperparameters (n_way, episodes, lr, ...)
└── roi_dataset.yaml

src/
├── models/         backbones.py (ResNet34/50/ViT-small factory), protonet.py
├── data/           episodic_dataset.py, transforms.py (train/eval/TTA transforms)
├── losses/         prototypical.py (shared ProtoNet prototype + distance logic)
└── utils/          seed.py

scripts/            all CLI entry points -- train/evaluate/summarize, ROI dataset creation,
                    few-shot, Grad-CAM, embedding plots, inference benchmarks (see Usage below)

results/            static snapshot of reported results; real checkpoints/metrics live under $DATA_ROOT
demo/               end-to-end inference + Grad-CAM + UMAP demo on a handful of test images
```

`baseline/scripts/legacy/` still exists **on disk** (gitignored, not tracked in this repo) for local
reference: superseded scripts operating on old dataset versions, one-off utilities, and drafts with bugs
that got fixed elsewhere. See "`legacy/`" below.

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
python scripts/build_roi_dataset.py
```

### 2. Train & evaluate baselines

```bash
# Full-image baseline (ResNet34, view MIX, seed 0)
python scripts/train.py --config config/baseline/resnet34.yaml --view MIX --seed 0
python scripts/evaluate.py --config config/baseline/resnet34.yaml --view MIX --seed 0

# ROI variant
python scripts/train.py --config config/baseline/resnet34_roi.yaml --view MIX --seed 0
python scripts/evaluate.py --config config/baseline/resnet34_roi.yaml --view MIX --seed 0
```

Swap the config for `resnet50[.yaml|_roi.yaml]` or `vit_small[.yaml|_roi.yaml]` to train the other
backbones — `scripts/train.py` and `scripts/evaluate.py` are shared across all six combinations. After
running all views/seeds, aggregate results with:

```bash
python scripts/summarize_results.py --backbone resnet34 --roi --tta
```

### 3. Few-shot learning (ProtoNet)

```bash
python scripts/train_protonet.py --view MIX --shot 5 --data_root "$DATA_ROOT"
python scripts/evaluate_protonet.py --view MIX --shot 5 --data_root "$DATA_ROOT" --model_root "$DATA_ROOT/models_fsl"
python scripts/evaluate_protonet.py --view MIX --shot 5 --data_root "$DATA_ROOT" --model_root "$DATA_ROOT/models_fsl" --tta 5

# Or run the full sweep (all views × shots, full-image and ROI):
./scripts/run_fewshot_experiments.sh
./scripts/run_eval_all.sh
./scripts/run_eval_tta_roi.sh

python scripts/summarize_fewshot_results.py --tta
```

`scripts/plot_embeddings.py --method {tsne,umap}` compares embedding spaces (ResNet34 full vs ROI vs
ProtoNet) for a given view/seed/shot (defaults: MIX, seed 0, 5-shot).

### 4. Grad-CAM

```bash
python scripts/generate_gradcam.py --roi --view MIX --seed 0
python scripts/plot_gradcam_single_model.py --roi --view MIX --seed 0
python scripts/plot_gradcam_comparison.py   # and the _originalsize / _grid / _random variants
```

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
python scripts/benchmark_inference_time.py --model resnet34            # full-image classifier
python scripts/benchmark_inference_time.py --model resnet34 --roi      # ROI classifier
python scripts/benchmark_inference_time.py --model protonet --shot 5   # full-image ProtoNet
python scripts/benchmark_inference_time.py --model protonet --roi --shot 5   # ROI ProtoNet
```

## Results

See [results/README.md](results/README.md) for a snapshot of reported accuracy per model/view and the
best checkpoint per view. Real checkpoints/metrics live under `$DATA_ROOT`, not in this repo.

## `legacy/`

`baseline/scripts/legacy/` (gitignored — not tracked in this repository, since it isn't part of the
reproducible pipeline) holds scripts kept only for local reference: ones that operate on superseded
dataset versions (`roi_dataset`/`roi_dataset_v3` instead of the current `roi_dataset_v4`), one-off
preprocessing utilities (`convert_images_to_png.py`), and superseded drafts (`plot_gradcam_full_mix.py`
loads the ROI checkpoint despite its name — fixed in `scripts/plot_gradcam_single_model.py`).


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
