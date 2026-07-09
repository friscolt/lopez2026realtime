"""
Train a supervised baseline classifier (ResNet34, ResNet50, or ViT-small) on full images or ROI crops.

Replaces the 6 near-identical train_{resnet34,resnet50,vit_small}_baseline[_roi].py scripts: the
backbone, image size, learning rate/optimizer, and dataset root now come from a config/baseline/*.yaml
file instead of being hardcoded per script.

Examples:
  python scripts/train.py --config config/baseline/resnet34_roi.yaml --view MIX --seed 0
  python scripts/train.py --config config/baseline/vit_small.yaml --view SEC --seed 1
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.transforms import build_eval_transform, build_train_transform  # noqa: E402
from src.models.backbones import build_classifier  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")


def build_optimizer(name, params, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name!r} (expected adam or adamw)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a config/baseline/*.yaml file")
    parser.add_argument("--view", required=True, choices=["SUR", "SEC", "MIX"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs (e.g. for a quick smoke test)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    backbone = cfg["backbone"]
    roi = cfg["roi"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    lr = cfg["lr"]
    optimizer_name = cfg["optimizer"]
    weight_decay = cfg["weight_decay"]
    patience = cfg["patience"]
    scheduler_factor = cfg["scheduler_factor"]
    scheduler_patience = cfg["scheduler_patience"]

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    view_root = f"{args.data_root}/roi_dataset_v4/{args.view}" if roi else f"{args.data_root}/{args.view}"
    train_dir = os.path.join(view_root, "train")
    test_dir = os.path.join(view_root, "test")

    model_dir_name = f"{backbone}_roi_baseline" if roi else f"{backbone}_baseline"
    model_dir = f"{args.data_root}/models/{model_dir_name}"
    os.makedirs(model_dir, exist_ok=True)

    file_prefix = f"{backbone}_roi_{args.view}_seed{args.seed}" if roi else f"{backbone}_{args.view}_seed{args.seed}"
    model_path = f"{model_dir}/{file_prefix}_best.pth"
    csv_path = f"{model_dir}/{file_prefix}_metrics.csv"
    json_path = f"{model_dir}/{file_prefix}_metrics.json"

    train_dataset = datasets.ImageFolder(train_dir, transform=build_train_transform(img_size))
    test_dataset = datasets.ImageFolder(test_dir, transform=build_eval_transform(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)

    model = build_classifier(backbone, num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        val_loss /= len(test_loader)

        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_path)
            print("Model saved:", model_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # Final evaluation with the best checkpoint
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=train_dataset.classes, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df.to_csv(csv_path)

    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\nMetrics saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
