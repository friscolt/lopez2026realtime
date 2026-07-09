"""
Evaluate a trained baseline classifier with 3-way flip TTA (original + horizontal flip + vertical flip).

Replaces the 6 near-identical evaluate_{resnet34,resnet50,vit_small}_[roi_]tta.py scripts.

Example:
  python scripts/evaluate.py --config config/baseline/resnet34_roi.yaml --view MIX --seed 0
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.transforms import build_eval_transform, tta_predictions  # noqa: E402
from src.models.backbones import build_classifier  # noqa: E402

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a config/baseline/*.yaml file")
    parser.add_argument("--view", required=True, choices=["SUR", "SEC", "MIX"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    backbone = cfg["backbone"]
    roi = cfg["roi"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dir = (
        f"{args.data_root}/roi_dataset_v4/{args.view}/test" if roi else f"{args.data_root}/{args.view}/test"
    )

    model_dir_name = f"{backbone}_roi_baseline" if roi else f"{backbone}_baseline"
    model_dir = f"{args.data_root}/models/{model_dir_name}"

    file_prefix = f"{backbone}_roi_{args.view}_seed{args.seed}" if roi else f"{backbone}_{args.view}_seed{args.seed}"
    model_path = f"{model_dir}/{file_prefix}_best.pth"
    csv_path = f"{model_dir}/{file_prefix}_tta_metrics.csv"
    json_path = f"{model_dir}/{file_prefix}_tta_metrics.json"

    dataset = datasets.ImageFolder(test_dir, transform=build_eval_transform(img_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(dataset.classes)

    model = build_classifier(backbone, num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)

            preds = tta_predictions(model, images)
            predicted = torch.argmax(preds, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=dataset.classes, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df.to_csv(csv_path)

    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)

    print("Metrics saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
