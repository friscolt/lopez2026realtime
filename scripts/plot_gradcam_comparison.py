# Work!


import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models, transforms, datasets
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# ==================================
# CONFIG
# ==================================

FULL_DATASET = f"{DATA_ROOT}/MIX/test"
ROI_DATASET = f"{DATA_ROOT}/roi_dataset_v4/MIX/test"

BASELINE_MODEL = f"{DATA_ROOT}/models/resnet34_baseline/resnet34_MIX_seed0_best.pth"
ROI_MODEL = f"{DATA_ROOT}/models/resnet34_roi_baseline/resnet34_roi_MIX_seed0_best.pth"

OUTPUT_DIR = f"{DATA_ROOT}/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224

# ==================================
# DATASET
# ==================================

dataset = datasets.ImageFolder(FULL_DATASET)

classes = dataset.classes
num_classes = len(classes)

# ==================================
# MODELS
# ==================================

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

# ==================================
# TRANSFORM
# ==================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==================================
# SELECT SAME IMAGE PER CLASS
# ==================================

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

# ==================================
# FIGURE
# ==================================

fig, axes = plt.subplots(len(samples), 4, figsize=(12, 3*len(samples)))

for i,(cls, full_path, roi_path) in enumerate(samples):

    # ---------- FULL IMAGE ----------
    
    full_img = Image.open(full_path).convert("RGB")
    full_tensor = transform(full_img).unsqueeze(0).cuda()

    full_rgb = np.array(full_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    cam_full = baseline_cam(input_tensor=full_tensor)[0]
    cam_full_img = show_cam_on_image(full_rgb, cam_full, use_rgb=True)

    # ---------- ROI IMAGE ----------
    
    roi_img = Image.open(roi_path).convert("RGB")
    roi_tensor = transform(roi_img).unsqueeze(0).cuda()

    roi_rgb = np.array(roi_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    cam_roi = roi_cam(input_tensor=roi_tensor)[0]
    cam_roi_img = show_cam_on_image(roi_rgb, cam_roi, use_rgb=True)

    # ---------- PLOT ----------
    
    axes[i,0].imshow(full_rgb)
    axes[i,0].set_title("Full Image")
    axes[i,0].axis("off")

    axes[i,1].imshow(cam_full_img)
    axes[i,1].set_title("Full + GradCAM")
    axes[i,1].axis("off")

    axes[i,2].imshow(roi_rgb)
    axes[i,2].set_title("ROI")
    axes[i,2].axis("off")

    axes[i,3].imshow(cam_roi_img)
    axes[i,3].set_title("ROI + GradCAM")
    axes[i,3].axis("off")

plt.tight_layout()

save_path = f"{OUTPUT_DIR}/gradcam_full_vs_roi_mix.png"

plt.savefig(save_path, dpi=300)

print("Saved figure:", save_path)