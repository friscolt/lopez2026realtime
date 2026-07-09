"""
Measure inference latency (ms/image for the ResNet34 classifier, ms/query for ProtoNet) across
SUR/SEC/MIX.

Replaces the 4 time/test_time_*.py scripts, which had two bugs fixed here: all four hardcoded
IMG_SIZE=256 regardless of what the model was actually trained/evaluated at (384 for the ResNet34
full-image baseline, 224 for ROI and for ProtoNet); and test_time_FSL.py / test_time_resnet34_full_images_fewshot.py
were two independent, functionally-identical implementations of the same "ProtoNet on full images"
benchmark. This version also adds a `protonet --roi` mode (timing the ProtoNet-ROI model), which
didn't exist before but is the best-performing few-shot model.

Example:
  python scripts/benchmark_inference_time.py --model resnet34 --roi
  python scripts/benchmark_inference_time.py --model protonet --shot 5
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.episodic_dataset import EpisodicDataset  # noqa: E402
from src.data.transforms import build_eval_transform  # noqa: E402
from src.models.backbones import build_classifier  # noqa: E402
from src.models.protonet import ProtoNet  # noqa: E402

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")
VIEWS = ["MIX", "SEC", "SUR"]
NUM_CLASSES = 6

N_WAY = 6
Q_QUERY = 15
EPISODES = 300


def measure_classifier(model, loader, device):
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            _ = model(images.to(device))
            if i >= 5:
                break

    total_time, total_images = 0.0, 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start
            total_images += images.size(0)

    return (total_time / total_images) * 1000  # ms/img


def measure_protonet(model, loader, device):
    with torch.no_grad():
        for i, batch in enumerate(loader):
            sx, sy, qx, _ = batch
            _ = model(sx.to(device))
            _ = model(qx.to(device))
            if i >= 5:
                break

    total_time, total_queries = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            sx, sy, qx, _ = batch
            sx, qx = sx.to(device), qx.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(sx)
            _ = model(qx)
            if device.type == "cuda":
                torch.cuda.synchronize()

            total_time += time.time() - start
            total_queries += qx.size(0)

    return (total_time / total_queries) * 1000  # ms/query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet34", "protonet"])
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="ResNet34 checkpoint seed")
    parser.add_argument("--shot", type=int, default=1, help="ProtoNet checkpoint shot count")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224 if (args.roi or args.model == "protonet") else 384
    transform = build_eval_transform(img_size)

    results = {}

    for view in VIEWS:
        print(f"\nRunning {view}")

        if args.model == "resnet34":
            data_dir = f"{args.data_root}/roi_dataset_v4/{view}/test" if args.roi else f"{args.data_root}/{view}/test"
            model_dir_name = "resnet34_roi_baseline" if args.roi else "resnet34_baseline"
            file_prefix = f"resnet34_roi_{view}_seed{args.seed}" if args.roi else f"resnet34_{view}_seed{args.seed}"
            model_path = f"{args.data_root}/models/{model_dir_name}/{file_prefix}_best.pth"

            dataset = datasets.ImageFolder(data_dir, transform=transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            model = build_classifier("resnet34", NUM_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device).eval()

            ms = measure_classifier(model, loader, device)
            unit = "ms/img"

        else:
            data_dir = f"{args.data_root}/roi_dataset_v4/{view}/test" if args.roi else f"{args.data_root}/{view}/test"
            model_variant = "protonet_resnet34_roi" if args.roi else "protonet_resnet34_full"
            model_path = f"{args.data_root}/models_fsl/{model_variant}/protonet_resnet34/{view}/{args.shot}shot/model.pth"

            dataset = EpisodicDataset(root=data_dir, transform=transform, n_way=N_WAY, k_shot=args.shot, q_query=Q_QUERY, episodes=EPISODES)
            loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4)

            model = ProtoNet()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device).eval()

            ms = measure_protonet(model, loader, device)
            unit = "ms/query"

        print(f"{view}: {ms:.2f} {unit}")
        results[view] = ms

    print(f"\n=== FINAL RESULTS ({unit}) ===")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    main()
