import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# ==================================================
# CONFIG
# ==================================================

DATASETS = ["SUR", "SEC", "MIX"]
SPLITS = ["train", "test"]

INPUT_ROOT = f"{DATA_ROOT}"
OUTPUT_ROOT = f"{DATA_ROOT}/roi_dataset_v4"

PADDING = 20

os.makedirs(OUTPUT_ROOT, exist_ok=True)

stats = []
errors = []

# ==================================================
# FUNCTION: GET BOUNDING BOX
# ==================================================

def get_bbox(mask):

    coords = np.column_stack(np.where(mask > 0))

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return x_min, y_min, x_max, y_max


# ==================================================
# MAIN LOOP
# ==================================================

for dataset in DATASETS:

    print("\nProcessing dataset:", dataset)

    for split in SPLITS:

        split_path = os.path.join(INPUT_ROOT, dataset, split)

        classes = sorted(os.listdir(split_path))

        for cls in classes:

            class_path = os.path.join(split_path, cls)
            mask_path = os.path.join(class_path, "mask")

            if not os.path.isdir(mask_path):
                continue

            output_class = os.path.join(
                OUTPUT_ROOT, dataset, split, cls
            )

            os.makedirs(output_class, exist_ok=True)

            images = sorted([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

            print(f"{dataset} {split} {cls}: {len(images)} images")

            for img_name in tqdm(images):

                img_file = os.path.join(class_path, img_name)

                base = os.path.splitext(img_name)[0]

                # --------------------------------------------------
                # FIND MASK WITH ANY EXTENSION
                # --------------------------------------------------

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

                # --------------------------------------------------
                # IF MASK MISSING → COPY ORIGINAL
                # --------------------------------------------------

                if mask_file is None:

                    errors.append(img_name)

                    save_path = os.path.join(output_class, base + ".png")

                    cv2.imwrite(save_path, img)

                    continue

                mask = cv2.imread(mask_file, 0)

                bbox = get_bbox(mask)

                # --------------------------------------------------
                # IF MASK EMPTY → COPY ORIGINAL
                # --------------------------------------------------

                if bbox is None:

                    errors.append(img_name)

                    save_path = os.path.join(output_class, base + ".png")

                    cv2.imwrite(save_path, img)

                    continue

                x1, y1, x2, y2 = bbox

                # --------------------------------------------------
                # APPLY PADDING
                # --------------------------------------------------

                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(img.shape[1], x2 + PADDING)
                y2 = min(img.shape[0], y2 + PADDING)

                roi = img[y1:y2, x1:x2]

                save_path = os.path.join(output_class, base + ".png")

                cv2.imwrite(save_path, roi)

                h, w, _ = roi.shape

                stats.append({

                    "dataset": dataset,
                    "split": split,
                    "class": cls,
                    "width": w,
                    "height": h,
                    "area": w * h

                })


# ==================================================
# SAVE STATS
# ==================================================

df = pd.DataFrame(stats)

stats_path = os.path.join(OUTPUT_ROOT, "roi_stats.csv")

df.to_csv(stats_path, index=False)

print("\nSaved ROI stats:", stats_path)


# ==================================================
# SUMMARY
# ==================================================

summary = df.groupby("dataset").agg({

    "width": ["mean", "min", "max"],
    "height": ["mean", "min", "max"],
    "area": ["mean", "min", "max"]

})

summary_path = os.path.join(OUTPUT_ROOT, "roi_summary.csv")

summary.to_csv(summary_path)

print("\nSaved summary:", summary_path)


# ==================================================
# FINAL REPORT
# ==================================================

print("\nROI dataset created at:", OUTPUT_ROOT)

print("Mask issues detected:", len(errors))