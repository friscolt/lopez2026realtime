"""
Build the ROI-cropped dataset from raw images + binary masks, using config/roi_dataset.yaml.

= create_roi_dataset_v4.py, unchanged logic, now reading settings (datasets, splits, padding,
output directory name) from a config file instead of hardcoded constants.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "roi_dataset.yaml")


def get_bbox(mask):
    coords = np.column_stack(np.where(mask > 0))

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return x_min, y_min, x_max, y_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    datasets_list = cfg["datasets"]
    splits = cfg["splits"]
    padding = cfg["padding"]

    input_root = args.data_root
    output_root = f"{args.data_root}/{cfg['output_dirname']}"
    os.makedirs(output_root, exist_ok=True)

    stats = []
    errors = []

    for dataset in datasets_list:
        print("\nProcessing dataset:", dataset)

        for split in splits:
            split_path = os.path.join(input_root, dataset, split)
            classes = sorted(os.listdir(split_path))

            for cls in classes:
                class_path = os.path.join(split_path, cls)
                mask_path = os.path.join(class_path, "mask")

                if not os.path.isdir(mask_path):
                    continue

                output_class = os.path.join(output_root, dataset, split, cls)
                os.makedirs(output_class, exist_ok=True)

                images = sorted([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])

                print(f"{dataset} {split} {cls}: {len(images)} images")

                for img_name in tqdm(images):
                    img_file = os.path.join(class_path, img_name)
                    base = os.path.splitext(img_name)[0]

                    mask_png = os.path.join(mask_path, base + ".png")
                    mask_jpg = os.path.join(mask_path, base + ".jpg")
                    mask_jpeg = os.path.join(mask_path, base + ".jpeg")

                    if os.path.exists(mask_png):
                        mask_file = mask_png
                    elif os.path.exists(mask_jpg):
                        mask_file = mask_jpg
                    elif os.path.exists(mask_jpeg):
                        mask_file = mask_jpeg
                    else:
                        mask_file = None

                    img = cv2.imread(img_file)

                    if mask_file is None:
                        errors.append(img_name)
                        cv2.imwrite(os.path.join(output_class, base + ".png"), img)
                        continue

                    mask = cv2.imread(mask_file, 0)
                    bbox = get_bbox(mask)

                    if bbox is None:
                        errors.append(img_name)
                        cv2.imwrite(os.path.join(output_class, base + ".png"), img)
                        continue

                    x1, y1, x2, y2 = bbox
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(img.shape[1], x2 + padding)
                    y2 = min(img.shape[0], y2 + padding)

                    roi = img[y1:y2, x1:x2]
                    save_path = os.path.join(output_class, base + ".png")
                    cv2.imwrite(save_path, roi)

                    h, w, _ = roi.shape
                    stats.append({"dataset": dataset, "split": split, "class": cls, "width": w, "height": h, "area": w * h})

    df = pd.DataFrame(stats)
    stats_path = os.path.join(output_root, "roi_stats.csv")
    df.to_csv(stats_path, index=False)
    print("\nSaved ROI stats:", stats_path)

    summary = df.groupby("dataset").agg({"width": ["mean", "min", "max"], "height": ["mean", "min", "max"], "area": ["mean", "min", "max"]})
    summary_path = os.path.join(output_root, "roi_summary.csv")
    summary.to_csv(summary_path)
    print("\nSaved summary:", summary_path)

    print("\nROI dataset created at:", output_root)
    print("Mask issues detected:", len(errors))


if __name__ == "__main__":
    main()
