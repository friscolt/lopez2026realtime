import os
import cv2
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# ==========================================
# CONFIG
# ==========================================

ORIGINAL_ROOT = f"{DATA_ROOT}/MIX"
ROI_ROOT = f"{DATA_ROOT}/roi_dataset_v4/MIX"

SPLITS = ["train", "test"]

THRESHOLD = 0.01  # diferencia mínima

total = 0
same = 0
changed = 0

examples_same = []

# ==========================================
# LOOP
# ==========================================

for split in SPLITS:

    original_split = os.path.join(ORIGINAL_ROOT, split)
    roi_split = os.path.join(ROI_ROOT, split)

    classes = sorted([
        d for d in os.listdir(original_split)
        if os.path.isdir(os.path.join(original_split, d))
])

    for cls in classes:

        original_class = os.path.join(original_split, cls)
        roi_class = os.path.join(roi_split, cls)

        for f in os.listdir(roi_class):

            roi_path = os.path.join(roi_class, f)

            base = os.path.splitext(f)[0]

            # buscar imagen original
            original_png = os.path.join(original_class, base + ".png")
            original_jpg = os.path.join(original_class, base + ".jpg")
            original_jpeg = os.path.join(original_class, base + ".jpeg")

            if os.path.exists(original_png):
                original_path = original_png
            elif os.path.exists(original_jpg):
                original_path = original_jpg
            elif os.path.exists(original_jpeg):
                original_path = original_jpeg
            else:
                continue

            img_orig = cv2.imread(original_path)
            img_roi = cv2.imread(roi_path)

            if img_orig is None or img_roi is None:
                continue

            # redimensionar para comparar
            img_orig = cv2.resize(img_orig, (224,224))
            img_roi = cv2.resize(img_roi, (224,224))

            diff = np.mean(np.abs(img_orig.astype("float") - img_roi.astype("float")))

            total += 1

            if diff < THRESHOLD:
                same += 1
                examples_same.append(original_path)
            else:
                changed += 1


# ==========================================
# RESULTS
# ==========================================

print("\n========== ROI VALIDATION ==========\n")

print("Total images:", total)
print("ROI changed:", changed)
print("ROI same as original:", same)

print("\nChange percentage:")

if total > 0:
    print(round(changed / total * 100,2), "%")

print("\nExamples where ROI did NOT change:")

for e in examples_same[:10]:
    print(e)