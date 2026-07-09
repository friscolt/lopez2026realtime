import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models, transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ============================
# CONFIG
# ============================

VIEW = "MIX"
SEED = 0

FULL_DATASET = f"{DATA_ROOT}/MIX/test"
ROI_DATASET = f"{DATA_ROOT}/roi_dataset_v3/MIX/test"

MODEL_PATH = f"{DATA_ROOT}/models/resnet34_roi_baseline/resnet34_roi_{VIEW}_seed{SEED}_best.pth"

OUTPUT_DIR = f"{DATA_ROOT}/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224

# ============================
# MODEL
# ============================

from torchvision import datasets

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

dataset = datasets.ImageFolder(FULL_DATASET)

dataset_classes = dataset.classes
num_classes = len(dataset_classes)

model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()
model.cuda()

target_layer = model.layer4[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

# ============================
# TRANSFORM
# ============================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

# ============================
# SAMPLE IMAGES
# ============================

samples = []

for cls in dataset_classes:

    class_path = os.path.join(FULL_DATASET, cls)

    imgs = os.listdir(class_path)

    samples.append(os.path.join(class_path, imgs[0]))

samples = samples[:6]

# ============================
# PROCESS
# ============================

fig, axes = plt.subplots(len(samples),4, figsize=(12,3*len(samples)))

for i, img_path in enumerate(samples):

    img = Image.open(img_path).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).cuda()

    rgb_img = np.array(img.resize((IMG_SIZE,IMG_SIZE))) / 255.0

    grayscale_cam = cam(input_tensor=img_tensor)[0]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # ROI image
    cls = os.path.basename(os.path.dirname(img_path))
    name = os.path.basename(img_path)

    roi_path = os.path.join(ROI_DATASET, cls, name)

    roi = Image.open(roi_path).convert("RGB")

    roi_tensor = transform(roi).unsqueeze(0).cuda()

    roi_rgb = np.array(roi.resize((IMG_SIZE,IMG_SIZE))) / 255.0

    roi_cam = cam(input_tensor=roi_tensor)[0]

    roi_cam_image = show_cam_on_image(roi_rgb, roi_cam, use_rgb=True)

    axes[i,0].imshow(rgb_img)
    axes[i,0].set_title("Full Image")

    axes[i,1].imshow(cam_image)
    axes[i,1].set_title("GradCAM Full")

    axes[i,2].imshow(roi_rgb)
    axes[i,2].set_title("ROI")

    axes[i,3].imshow(roi_cam_image)
    axes[i,3].set_title("GradCAM ROI")

    for j in range(4):
        axes[i,j].axis("off")

plt.tight_layout()

save_path = f"{OUTPUT_DIR}/gradcam_comparison.png"

plt.savefig(save_path, dpi=300)

print("Saved:", save_path)