import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torchvision.models as models
from models.protonet_resnet34 import ProtoNet

import umap

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# =====================
# CONFIG
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6

# =====================
# PATHS
# =====================
FULL_MODEL_PATH = f"{DATA_ROOT}/models/resnet34_baseline/resnet34_MIX_seed0_best.pth"
ROI_MODEL_PATH  = f"{DATA_ROOT}/models/resnet34_roi_baseline/resnet34_roi_MIX_seed0_best.pth"
PROTO_MODEL_PATH = f"{DATA_ROOT}/models_fsl/protonet_resnet34_roi/protonet_resnet34/MIX/5shot/model.pth"

# =====================
# DATA
# =====================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset_full = datasets.ImageFolder(f"{DATA_ROOT}/MIX/test", transform=transform)
dataset_roi  = datasets.ImageFolder(f"{DATA_ROOT}/roi_dataset_v4/MIX/test", transform=transform)

# =====================
# CHECK CLASS CONSISTENCY
# =====================
print("FULL mapping:", dataset_full.class_to_idx)
print("ROI mapping :", dataset_roi.class_to_idx)

assert dataset_full.class_to_idx == dataset_roi.class_to_idx, "Class mappings do not match!"

# =====================
# CLASS NAMES (CLEAN)
# =====================
def clean_name(name):
    return name.split("_")[-1]

class_names_raw = dataset_full.classes
class_names = [clean_name(c) for c in class_names_raw]

print("Class order used:", class_names)

# =====================
# LOADERS
# =====================
loader_full = DataLoader(dataset_full, batch_size=32, shuffle=False)
loader_roi  = DataLoader(dataset_roi, batch_size=32, shuffle=False)

# =====================
# UTILS
# =====================
def clean_state_dict(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def load_resnet(path):
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(512, NUM_CLASSES)

    state = clean_state_dict(torch.load(path, map_location=device))
    model.load_state_dict(state)

    model.fc = torch.nn.Identity()
    return model.to(device).eval()

# =====================
# LOAD MODELS
# =====================
model_full = load_resnet(FULL_MODEL_PATH)
model_roi  = load_resnet(ROI_MODEL_PATH)

model_proto = ProtoNet()
model_proto.load_state_dict(torch.load(PROTO_MODEL_PATH, map_location=device))
model_proto = model_proto.to(device).eval()

# =====================
# FEATURE EXTRACTION
# =====================
def extract_features(model, loader):
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model(x)
            feats.append(emb.cpu().numpy())
            labs.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labs)

feat_full, labels_full   = extract_features(model_full, loader_full)
feat_roi, labels_roi     = extract_features(model_roi, loader_roi)
feat_proto, labels_proto = extract_features(model_proto, loader_roi)

# =====================
# SANITY CHECKS
# =====================
print("\nSanity check labels:")
for i in range(5):
    print(f"{labels_full[i]} -> {class_names[int(labels_full[i])]}")

print("\nSamples per class:")
for i, name in enumerate(class_names):
    print(f"{name}: {np.sum(labels_full == i)}")

# =====================
# UMAP
# =====================
def run_umap(features):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(features)

emb_full  = run_umap(feat_full)
emb_roi   = run_umap(feat_roi)
emb_proto = run_umap(feat_proto)

# =====================
# COLORS (CONSISTENT)
# =====================

colors = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02"
]


# =====================
# PLOT
# =====================
fig, axes = plt.subplots(1, 3, figsize=(20,7))

titles = ["ResNet34 Full", "ResNet34 ROI", "ProtoNet (5-shot)"]
embeddings = [emb_full, emb_roi, emb_proto]
label_sets = [labels_full, labels_roi, labels_proto]

for ax, emb, title, lbls in zip(axes, embeddings, titles, label_sets):

    unique_classes = np.unique(lbls)

    for class_id in unique_classes:
        idx = lbls == class_id

        ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            color=colors[class_id],
            s=80,
            alpha=0.9,
            edgecolors='k',
            linewidths=0.4
        )

    ax.set_title(title, fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])

# =====================
# LEGEND (CLEAN + MATCHED)
# =====================
handles = []
for i, name in enumerate(class_names):
    handles.append(
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=name,
            markerfacecolor=colors[i],
            markeredgecolor='k',
            markersize=12
        )
    )

legend = fig.legend(
    handles=handles,
    loc='lower center',
    ncol=6,
    fontsize=14,
    frameon=True
)

legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.2)

plt.tight_layout(rect=[0, 0.12, 1, 1])

plt.savefig("umap_comparison_mix.png", dpi=300)

print("✅ Saved UMAP figure")