import os
import pandas as pd
import json

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# ===============================
# CONFIG
# ===============================

DATASETS = ["SUR", "SEC", "MIX"]

BASE_PATH = f"{DATA_ROOT}"

OUTPUT_DIR = f"{DATA_ROOT}/models/dataset_stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "dataset_image_counts.csv")
JSON_PATH = os.path.join(OUTPUT_DIR, "dataset_image_counts.json")

results = []

# ===============================
# COUNT IMAGES
# ===============================

for dataset in DATASETS:

    for split in ["train", "test"]:

        path = os.path.join(BASE_PATH, dataset, split)

        classes = sorted(os.listdir(path))

        for cls in classes:

            cls_path = os.path.join(path, cls)

            if not os.path.isdir(cls_path):
                continue

            images = [
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            count = len(images)

            results.append({
                "dataset": dataset,
                "split": split,
                "class": cls,
                "num_images": count
            })

# ===============================
# SAVE RESULTS
# ===============================

df = pd.DataFrame(results)

df.to_csv(CSV_PATH, index=False)

with open(JSON_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\nImage counts by class:\n")
print(df)

print("\nSaved files:")
print(CSV_PATH)
print(JSON_PATH)