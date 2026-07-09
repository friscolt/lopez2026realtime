import os
import json
import argparse
import torch
import torch.nn as nn
import pandas as pd

import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# ===============================
# ARGUMENTS
# ===============================

parser = argparse.ArgumentParser()

parser.add_argument("--view", required=True, choices=["SUR","SEC","MIX"])
parser.add_argument("--seed", type=int, required=True)

args = parser.parse_args()

VIEW = args.view
SEED = args.seed

# ===============================
# PATHS
# ===============================

TEST_DIR = f"{DATA_ROOT}/{VIEW}/test"

MODEL_PATH = f"{DATA_ROOT}/models/vit_small_baseline/vit_small_{VIEW}_seed{SEED}_best.pth"

OUTPUT_DIR = f"{DATA_ROOT}/models/vit_small_baseline"

CSV_PATH = f"{OUTPUT_DIR}/vit_small_{VIEW}_seed{SEED}_tta_metrics.csv"
JSON_PATH = f"{OUTPUT_DIR}/vit_small_{VIEW}_seed{SEED}_tta_metrics.json"

# ===============================
# SETTINGS
# ===============================

IMG_SIZE = 224
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# DATA
# ===============================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

NUM_CLASSES = len(dataset.classes)

print("Classes:", dataset.classes)

# ===============================
# MODEL
# ===============================

model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH))

model = model.to(DEVICE)
model.eval()

# ===============================
# TTA INFERENCE
# ===============================

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)

        # original
        pred1 = torch.softmax(model(images), dim=1)

        # horizontal flip
        pred2 = torch.softmax(
            model(torch.flip(images, dims=[3])), dim=1
        )

        # vertical flip
        pred3 = torch.softmax(
            model(torch.flip(images, dims=[2])), dim=1
        )

        preds = (pred1 + pred2 + pred3) / 3

        predicted = torch.argmax(preds, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

# ===============================
# METRICS
# ===============================

report = classification_report(
    y_true,
    y_pred,
    target_names=dataset.classes,
    output_dict=True
)

df = pd.DataFrame(report).transpose()

df.to_csv(CSV_PATH)

with open(JSON_PATH, "w") as f:
    json.dump(report, f, indent=4)

print("\nMetrics saved:")
print(CSV_PATH)
print(JSON_PATH)