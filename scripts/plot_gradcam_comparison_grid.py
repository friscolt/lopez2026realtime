import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torchvision import models, transforms, datasets
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# =====================================
# CONFIG
# =====================================

FULL_DATASET = f"{DATA_ROOT}/MIX/test"
ROI_DATASET = f"{DATA_ROOT}/roi_dataset_v4/MIX/test"

BASELINE_MODEL = f"{DATA_ROOT}/models/resnet34_baseline/resnet34_MIX_seed0_best.pth"
ROI_MODEL = f"{DATA_ROOT}/models/resnet34_roi_baseline/resnet34_roi_MIX_seed0_best.pth"

OUTPUT_DIR = f"{DATA_ROOT}/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZE = 224

# =====================================
# DATASET
# =====================================

dataset = datasets.ImageFolder(FULL_DATASET)

classes = dataset.classes
num_classes = len(classes)

print("Classes:", classes)

# =====================================
# MODELS
# =====================================

baseline_model = models.resnet34(weights=None)
baseline_model.fc = torch.nn.Linear(baseline_model.fc.in_features, num_classes)
baseline_model.load_state_dict(torch.load(BASELINE_MODEL))
baseline_model.eval().cuda()

roi_model = models.resnet34(weights=None)
roi_model.fc = torch.nn.Linear(roi_model.fc.in_features, num_classes)
roi_model.load_state_dict(torch.load(ROI_MODEL))
roi_model.eval().cuda()

baseline_cam = GradCAM(model=baseline_model, target_layers=[baseline_model.layer4[-1]])
roi_cam = GradCAM(model=roi_model, target_layers=[roi_model.layer4[-1]])

# =====================================
# TRANSFORM (ONLY FOR MODEL INPUT)
# =====================================

transform = transforms.Compose([
    transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
    transforms.ToTensor()
])

# =====================================
# SELECT SAME IMAGE PER CLASS
# =====================================

samples = []

for cls in classes:

    class_path = os.path.join(FULL_DATASET, cls)

    images = sorted([
        f for f in os.listdir(class_path)
        if f.lower().endswith((".png",".jpg",".jpeg"))
    ])

    img_name = images[0]

    full_path = os.path.join(class_path, img_name)

    base = os.path.splitext(img_name)[0]

    roi_path = os.path.join(ROI_DATASET, cls, base + ".png")

    samples.append((cls, full_path, roi_path))

# =====================================
# FIGURE (4 rows × 6 columns)
# =====================================

fig, axes = plt.subplots(4, len(samples), figsize=(3*len(samples), 12))

row_labels = [
    "Full Image",
    "Full + GradCAM",
    "ROI",
    "ROI + GradCAM"
]

for col,(cls, full_path, roi_path) in enumerate(samples):

    # ---------- FULL IMAGE ----------

    full_img = Image.open(full_path).convert("RGB")

    full_np = np.array(full_img)
    full_rgb = full_np.astype(np.float32) / 255

    input_tensor = transform(full_img).unsqueeze(0).cuda()

    cam_full = baseline_cam(input_tensor=input_tensor)[0]

    cam_full = cv2.resize(cam_full, (full_np.shape[1], full_np.shape[0]))

    cam_full_img = show_cam_on_image(full_rgb, cam_full, use_rgb=True)

    # ---------- ROI IMAGE ----------

    roi_img = Image.open(roi_path).convert("RGB")

    roi_np = np.array(roi_img)
    roi_rgb = roi_np.astype(np.float32) / 255

    roi_tensor = transform(roi_img).unsqueeze(0).cuda()

    cam_roi = roi_cam(input_tensor=roi_tensor)[0]

    cam_roi = cv2.resize(cam_roi, (roi_np.shape[1], roi_np.shape[0]))

    cam_roi_img = show_cam_on_image(roi_rgb, cam_roi, use_rgb=True)

    # ---------- PLOT ----------

    axes[0,col].imshow(full_np)
    axes[0,col].set_title(cls)
    axes[0,col].axis("off")

    axes[1,col].imshow(cam_full_img)
    axes[1,col].axis("off")

    axes[2,col].imshow(roi_np)
    axes[2,col].axis("off")

    axes[3,col].imshow(cam_roi_img)
    axes[3,col].axis("off")

# Row labels

for row in range(4):
    axes[row,0].set_ylabel(row_labels[row], fontsize=13)

plt.tight_layout()

save_path = f"{OUTPUT_DIR}/gradcam_full_vs_roi_grid.png"

plt.savefig(save_path, dpi=400, bbox_inches="tight")

print("Saved figure:", save_path)