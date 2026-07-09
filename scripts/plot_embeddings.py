"""
Compare embedding spaces (ResNet34 full, ResNet34 ROI, ProtoNet ROI) via 2D projection.

Replaces plot_tsne_compare.py + plot_umap_compare_clean.py: pass --method {tsne,umap}.

Example:
  python scripts/plot_embeddings.py --method umap --view MIX --seed 0 --shot 5
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbones import build_classifier  # noqa: E402
from src.models.protonet import ProtoNet  # noqa: E402

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")
NUM_CLASSES = 6

COLORS = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
]


def clean_name(name):
    return name.split("_")[-1]


def clean_state_dict(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_encoder(model_path, device):
    model = build_classifier("resnet34", NUM_CLASSES, pretrained=False)
    state = clean_state_dict(torch.load(model_path, map_location=device))
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    return model.to(device).eval()


def extract_features(model, loader, device):
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model(x)
            feats.append(emb.cpu().numpy())
            labs.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["tsne", "umap"])
    parser.add_argument("--view", default="MIX", choices=["SUR", "SEC", "MIX"])
    parser.add_argument("--seed", type=int, default=0, help="seed of the ResNet34 baseline checkpoints")
    parser.add_argument("--shot", type=int, default=5, help="shot count of the ProtoNet checkpoint")
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_model_path = f"{args.data_root}/models/resnet34_baseline/resnet34_{args.view}_seed{args.seed}_best.pth"
    roi_model_path = f"{args.data_root}/models/resnet34_roi_baseline/resnet34_roi_{args.view}_seed{args.seed}_best.pth"
    proto_model_path = f"{args.data_root}/models_fsl/protonet_resnet34_roi/protonet_resnet34/{args.view}/{args.shot}shot/model.pth"

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset_full = datasets.ImageFolder(f"{args.data_root}/{args.view}/test", transform=transform)
    dataset_roi = datasets.ImageFolder(f"{args.data_root}/roi_dataset_v4/{args.view}/test", transform=transform)

    assert dataset_full.class_to_idx == dataset_roi.class_to_idx, "Class mappings do not match!"

    class_names = [clean_name(c) for c in dataset_full.classes]
    print("Class order used:", class_names)

    loader_full = DataLoader(dataset_full, batch_size=32, shuffle=False)
    loader_roi = DataLoader(dataset_roi, batch_size=32, shuffle=False)

    model_full = load_encoder(full_model_path, device)
    model_roi = load_encoder(roi_model_path, device)

    model_proto = ProtoNet()
    model_proto.load_state_dict(torch.load(proto_model_path, map_location=device))
    model_proto = model_proto.to(device).eval()

    feat_full, labels_full = extract_features(model_full, loader_full, device)
    feat_roi, labels_roi = extract_features(model_roi, loader_roi, device)
    feat_proto, labels_proto = extract_features(model_proto, loader_roi, device)

    if args.method == "tsne":
        from sklearn.manifold import TSNE

        def reduce(features):
            return TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42).fit_transform(features)
    else:
        import umap

        def reduce(features):
            return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(features)

    emb_full = reduce(feat_full)
    emb_roi = reduce(feat_roi)
    emb_proto = reduce(feat_proto)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    titles = ["ResNet34 Full", "ResNet34 ROI", "ProtoNet (ROI)"]
    embeddings = [emb_full, emb_roi, emb_proto]
    label_sets = [labels_full, labels_roi, labels_proto]

    for ax, emb, title, lbls in zip(axes, embeddings, titles, label_sets):
        for class_id in np.unique(lbls):
            idx = lbls == class_id
            ax.scatter(emb[idx, 0], emb[idx, 1], color=COLORS[class_id], s=80, alpha=0.9, edgecolors="k", linewidths=0.4)
        ax.set_title(title, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=COLORS[i], markeredgecolor="k", markersize=12)
        for i, name in enumerate(class_names)
    ]
    legend = fig.legend(handles=handles, loc="lower center", ncol=len(class_names), fontsize=14, frameon=True)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.2)

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    save_path = f"{args.method}_comparison_{args.view.lower()}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {args.method.upper()} figure:", save_path)


if __name__ == "__main__":
    main()
