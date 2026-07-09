import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# ===============================
# CONFIG
# ===============================

ROOT = f"{DATA_ROOT}/roi_dataset/MIX"

SPLITS = ["train","test"]

N_SAMPLES = 20

OUTPUT_FIG = f"{DATA_ROOT}/roi_dataset/mix_roi_samples.png"

# ===============================
# COLLECT IMAGES
# ===============================

images = []

for split in SPLITS:

    split_path = os.path.join(ROOT,split)

    classes = os.listdir(split_path)

    for cls in classes:

        class_path = os.path.join(split_path,cls)

        files = [
            f for f in os.listdir(class_path)
            if f.endswith((".png",".jpg",".jpeg"))
        ]

        for f in files:

            images.append({
                "path": os.path.join(class_path,f),
                "class": cls
            })

print("Total MIX images:",len(images))

# ===============================
# SAMPLE RANDOM ROIS
# ===============================

samples = random.sample(images,N_SAMPLES)

# ===============================
# PLOT GRID
# ===============================

fig, axes = plt.subplots(4,5,figsize=(15,12))

axes = axes.flatten()

sizes = []

for i,s in enumerate(samples):

    img = cv2.imread(s["path"])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    h,w,_ = img.shape

    sizes.append((w,h))

    axes[i].imshow(img)
    axes[i].set_title(f"{s['class']}\n{w}x{h}",fontsize=9)
    axes[i].axis("off")

plt.tight_layout()

plt.savefig(OUTPUT_FIG,dpi=300)

print("\nSaved visualization:")
print(OUTPUT_FIG)

plt.show()

# ===============================
# SIZE STATISTICS
# ===============================

sizes = np.array(sizes)

widths = sizes[:,0]
heights = sizes[:,1]
areas = widths*heights

stats = {

    "mean_width": np.mean(widths),
    "mean_height": np.mean(heights),
    "mean_area": np.mean(areas),

    "min_width": np.min(widths),
    "max_width": np.max(widths),

    "min_height": np.min(heights),
    "max_height": np.max(heights),

    "min_area": np.min(areas),
    "max_area": np.max(areas)

}

print("\nROI size statistics (sampled):")

for k,v in stats.items():

    print(f"{k}: {v:.2f}")

df = pd.DataFrame([stats])

df.to_csv(f"{DATA_ROOT}/roi_dataset/mix_roi_size_stats_sample.csv",index=False)

print("\nSaved stats:")
print(f"{DATA_ROOT}/roi_dataset/mix_roi_size_stats_sample.csv")