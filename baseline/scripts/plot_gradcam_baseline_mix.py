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

# ==============================
# CONFIG
# ==============================

DATASET_PATH = f"{DATA_ROOT}/MIX/test"

MODEL_PATH = f"{DATA_ROOT}/models/resnet34_baseline/resnet34_MIX_seed0_best.pth"

OUTPUT_DIR = f"{DATA_ROOT}/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224

# ==============================
# DATASET
# ==============================

dataset = datasets.ImageFolder(DATASET_PATH)

classes = dataset.classes
num_classes = len(classes)

print("Classes:", classes)

# ==============================
# MODEL
# ==============================

model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH))

model.eval().cuda()

target_layer = model.layer4[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==============================
# SELECT ONE IMAGE PER CLASS
# ==============================

samples = []

for cls in classes:

    class_path = os.path.join(DATASET_PATH, cls)

    images = sorted(os.listdir(class_path))

    samples.append(os.path.join(class_path, images[0]))

# ==============================
# FIGURE
# ==============================

fig, axes = plt.subplots(len(samples), 2, figsize=(8, 3*len(samples)))

for i, img_path in enumerate(samples):

    img = Image.open(img_path).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).cuda()

    rgb_img = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    grayscale_cam = cam(input_tensor=img_tensor)[0]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    label = os.path.basename(os.path.dirname(img_path))

    # ORIGINAL IMAGE
    axes[i,0].imshow(rgb_img)
    axes[i,0].set_title(f"{label} - RGB")
    axes[i,0].axis("off")

    # GRADCAM
    axes[i,1].imshow(cam_image)
    axes[i,1].set_title("RGB + GradCAM")
    axes[i,1].axis("off")

plt.tight_layout()

save_path = f"{OUTPUT_DIR}/gradcam_baseline_mix.png"

plt.savefig(save_path, dpi=300)

print("Saved figure:", save_path)