"""
End-to-end demo: pick N test images per class, run inference with the
ResNet34 baseline (full image and ROI), visualize Grad-CAM for those
examples, and plot a UMAP of the full test set (ResNet34 full, ResNet34
ROI, ProtoNet ROI) with the chosen examples highlighted.

Requires trained checkpoints already present under $DATA_ROOT:
  models/resnet34_baseline/resnet34_{view}_seed{seed}_best.pth
  models/resnet34_roi_baseline/resnet34_roi_{view}_seed{seed}_best.pth
  models_fsl/protonet_resnet34_roi/protonet_resnet34/{view}/{shot}shot/model.pth

Example:
  python demo/run_inference_demo.py --view MIX --seed 0 --shot 5 --n_per_class 1
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models as tvmodels, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import umap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from src.models.protonet import ProtoNet  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

parser = argparse.ArgumentParser()
parser.add_argument("--view", default="MIX", choices=["SUR", "SEC", "MIX"])
parser.add_argument("--seed", type=int, default=0, help="seed of the ResNet34 baseline checkpoints")
parser.add_argument("--shot", type=int, default=5, help="shot count of the ProtoNet checkpoint")
parser.add_argument("--n_per_class", type=int, default=1, help="number of test examples per class to run inference/Grad-CAM on")
parser.add_argument("--data_root", default=DATA_ROOT)
parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
print("Device:", DEVICE)

FULL_MODEL_PATH = f"{args.data_root}/models/resnet34_baseline/resnet34_{args.view}_seed{args.seed}_best.pth"
ROI_MODEL_PATH = f"{args.data_root}/models/resnet34_roi_baseline/resnet34_roi_{args.view}_seed{args.seed}_best.pth"
PROTO_MODEL_PATH = f"{args.data_root}/models_fsl/protonet_resnet34_roi/protonet_resnet34/{args.view}/{args.shot}shot/model.pth"

FULL_TEST_DIR = f"{args.data_root}/{args.view}/test"
ROI_TEST_DIR = f"{args.data_root}/roi_dataset_v4/{args.view}/test"

os.makedirs(args.out_dir, exist_ok=True)

# ------------------------------------------------------------------
# class mapping (must match ImageFolder's own sorted class_to_idx,
# the same convention used when the checkpoints were trained)
# ------------------------------------------------------------------
full_dataset_meta = datasets.ImageFolder(FULL_TEST_DIR)
idx_to_class = {v: k for k, v in full_dataset_meta.class_to_idx.items()}
classes = full_dataset_meta.classes
print("Classes:", classes)

# ------------------------------------------------------------------
# pick N test images per class (deterministic: first files, sorted)
# ------------------------------------------------------------------
examples = []
for cls in classes:
    cls_dir = os.path.join(FULL_TEST_DIR, cls)
    files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    for fname in files[: args.n_per_class]:
        examples.append({
            "class": cls, "filename": fname,
            "full_path": os.path.join(FULL_TEST_DIR, cls, fname),
            "roi_path": os.path.join(ROI_TEST_DIR, cls, fname),
        })

print(f"\nSelected {len(examples)} examples ({args.n_per_class} per class):")
for e in examples:
    print(" ", e["class"], "->", e["filename"])

# ------------------------------------------------------------------
# models
# ------------------------------------------------------------------
def load_resnet34(path):
    m = tvmodels.resnet34(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m

full_model = load_resnet34(FULL_MODEL_PATH)
roi_model = load_resnet34(ROI_MODEL_PATH)

proto_model = ProtoNet()
proto_model.load_state_dict(torch.load(PROTO_MODEL_PATH, map_location=DEVICE))
proto_model.to(DEVICE).eval()

# transforms MATCHING how each model was actually trained (Resize + ToTensor, no normalize)
transform_full = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])
transform_roi = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# ------------------------------------------------------------------
# 1) INFERENCE on the chosen examples (full model + ROI model)
# ------------------------------------------------------------------
print("\n=== INFERENCE ===")
results = []
with torch.no_grad():
    for e in examples:
        img_full = Image.open(e["full_path"]).convert("RGB")
        x_full = transform_full(img_full).unsqueeze(0).to(DEVICE)
        out_full = torch.softmax(full_model(x_full), dim=1)[0]
        pred_full = idx_to_class[int(out_full.argmax())]
        conf_full = float(out_full.max())

        img_roi = Image.open(e["roi_path"]).convert("RGB")
        x_roi = transform_roi(img_roi).unsqueeze(0).to(DEVICE)
        out_roi = torch.softmax(roi_model(x_roi), dim=1)[0]
        pred_roi = idx_to_class[int(out_roi.argmax())]
        conf_roi = float(out_roi.max())

        row = {
            "true_class": e["class"], "file": e["filename"],
            "pred_full": pred_full, "conf_full": round(conf_full, 4), "correct_full": pred_full == e["class"],
            "pred_roi": pred_roi, "conf_roi": round(conf_roi, 4), "correct_roi": pred_roi == e["class"],
        }
        results.append(row)
        print(f"{e['class']:20s} | full: {pred_full:20s} ({conf_full:.2%}) {'OK' if row['correct_full'] else 'X'}"
              f"   | roi: {pred_roi:20s} ({conf_roi:.2%}) {'OK' if row['correct_roi'] else 'X'}")

with open(os.path.join(args.out_dir, "inference_results.json"), "w") as f:
    json.dump(results, f, indent=2)

acc_full = sum(r["correct_full"] for r in results) / len(results)
acc_roi = sum(r["correct_roi"] for r in results) / len(results)
print(f"\nAccuracy on these {len(results)} examples -> full: {acc_full:.1%} | roi: {acc_roi:.1%}")

# ------------------------------------------------------------------
# 2) GRAD-CAM for the chosen examples, both models, side by side
# ------------------------------------------------------------------
print("\n=== GRAD-CAM ===")

def gradcam_overlay(model, img_path, transform, img_size):
    img = Image.open(img_path).convert("RGB")
    rgb_img = np.array(img.resize((img_size, img_size))).astype(np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_class = model(input_tensor).argmax(dim=1).item()

    cam = GradCAM(model=model, target_layers=[model.layer4])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization, idx_to_class[pred_class]

fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 9), squeeze=False)

for i, e in enumerate(examples):
    vis_full, pred_full = gradcam_overlay(full_model, e["full_path"], transform_full, 384)
    axes[0, i].imshow(vis_full)
    axes[0, i].set_title(f"{e['class']}\nfull -> {pred_full}", fontsize=9)
    axes[0, i].axis("off")

    vis_roi, pred_roi = gradcam_overlay(roi_model, e["roi_path"], transform_roi, 224)
    axes[1, i].imshow(vis_roi)
    axes[1, i].set_title(f"roi -> {pred_roi}", fontsize=9)
    axes[1, i].axis("off")

plt.tight_layout()
gradcam_path = os.path.join(args.out_dir, "gradcam_examples.png")
plt.savefig(gradcam_path, dpi=150)
plt.close()
print("Saved:", gradcam_path)

# ------------------------------------------------------------------
# 3) UMAP over the FULL test set, 3 models, with the chosen examples
#    highlighted (UMAP needs enough points for its neighbor graph --
#    a handful of points alone would be degenerate).
# ------------------------------------------------------------------
print("\n=== UMAP (full test set, chosen examples highlighted) ===")

dataset_full = datasets.ImageFolder(FULL_TEST_DIR, transform=transform_full)
dataset_roi = datasets.ImageFolder(ROI_TEST_DIR, transform=transform_roi)
assert dataset_full.class_to_idx == dataset_roi.class_to_idx

loader_full = DataLoader(dataset_full, batch_size=16, shuffle=False)
loader_roi = DataLoader(dataset_roi, batch_size=16, shuffle=False)

def as_encoder(model):
    model.fc = torch.nn.Identity()
    return model

full_encoder = as_encoder(load_resnet34(FULL_MODEL_PATH))
roi_encoder = as_encoder(load_resnet34(ROI_MODEL_PATH))

def extract_features(model, loader):
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            emb = model(x)
            feats.append(emb.cpu().numpy())
            labs.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labs)

feat_full, labels_full = extract_features(full_encoder, loader_full)
feat_roi, labels_roi = extract_features(roi_encoder, loader_roi)
feat_proto, labels_proto = extract_features(proto_model, loader_roi)

def run_umap(features):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(features)

emb_full = run_umap(feat_full)
emb_roi = run_umap(feat_roi)
emb_proto = run_umap(feat_proto)

full_samples = [s[0] for s in dataset_full.samples]
roi_samples = [s[0] for s in dataset_roi.samples]
highlight_idx_full = [full_samples.index(e["full_path"]) for e in examples]
highlight_idx_roi = [roi_samples.index(e["roi_path"]) for e in examples]

palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
           "#a6761d", "#666666", "#1f78b4", "#33a02c"]
colors = [palette[i % len(palette)] for i in range(len(classes))]

def clean_name(name):
    return name.split("_")[-1]

class_names = [clean_name(c) for c in classes]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
titles = ["ResNet34 Full", "ResNet34 ROI", "ProtoNet (ROI)"]
embeddings = [emb_full, emb_roi, emb_proto]
label_sets = [labels_full, labels_roi, labels_proto]
highlight_sets = [highlight_idx_full, highlight_idx_roi, highlight_idx_roi]

for ax, emb, title, lbls, hi in zip(axes, embeddings, titles, label_sets, highlight_sets):
    for class_id in np.unique(lbls):
        idx = lbls == class_id
        ax.scatter(emb[idx, 0], emb[idx, 1], color=colors[class_id], s=50, alpha=0.5, edgecolors="k", linewidths=0.3)
    for i in hi:
        ax.scatter(emb[i, 0], emb[i, 1], color=colors[lbls[i]],
                   s=260, marker="*", edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

handles = [plt.Line2D([0], [0], marker="o", color="w", label=n, markerfacecolor=colors[i], markeredgecolor="k", markersize=10)
           for i, n in enumerate(class_names)]
handles.append(plt.Line2D([0], [0], marker="*", color="w", label="highlighted example",
                           markerfacecolor="grey", markeredgecolor="k", markersize=16))
fig.legend(handles=handles, loc="lower center", ncol=len(class_names) + 1, fontsize=11, frameon=True)
plt.tight_layout(rect=[0, 0.1, 1, 1])
umap_path = os.path.join(args.out_dir, "umap_highlighted.png")
plt.savefig(umap_path, dpi=150)
plt.close()
print("Saved:", umap_path)

print("\nDONE. Outputs in:", args.out_dir)
