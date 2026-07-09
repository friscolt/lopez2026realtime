import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models, transforms, datasets
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# -----------------------------
# CONFIG
# -----------------------------

VIEW = "MIX"
SEED = 0

FULL_DATASET = f"{DATA_ROOT}/MIX/test"
ROI_DATASET = f"{DATA_ROOT}/roi_dataset_v3/MIX/test"

MODEL_PATH = f"{DATA_ROOT}/models/resnet34_roi_baseline/resnet34_roi_{VIEW}_seed{SEED}_best.pth"

OUTPUT_DIR = f"{DATA_ROOT}/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224

# -----------------------------
# LOAD DATASET
# -----------------------------

dataset = datasets.ImageFolder(FULL_DATASET)
classes = dataset.classes
num_classes = len(classes)

# -----------------------------
# MODEL
# -----------------------------

model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval().cuda()

target_layer = model.layer4[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

# -----------------------------
# TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -----------------------------
# SELECT SAMPLE IMAGES
# -----------------------------

samples = []

for cls in classes:
    path = os.path.join(FULL_DATASET, cls)
    imgs = os.listdir(path)
    samples.append(os.path.join(path, imgs[0]))

samples = samples[:6]

# -----------------------------
# FIGURE
# -----------------------------

fig, axes = plt.subplots(len(samples), 5, figsize=(15, 3*len(samples)))

for i, img_path in enumerate(samples):

    cls = os.path.basename(os.path.dirname(img_path))
    name = os.path.basename(img_path)

    # FULL IMAGE
    img = Image.open(img_path).convert("RGB")
    full = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    img_tensor = transform(img).unsqueeze(0).cuda()

    # GRADCAM
    grayscale_cam = cam(input_tensor=img_tensor)[0]
    cam_image = show_cam_on_image(full, grayscale_cam, use_rgb=True)

    # ROI IMAGE
    roi_path = os.path.join(ROI_DATASET, cls, name)

    roi_img = Image.open(roi_path).convert("RGB")
    roi = np.array(roi_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    roi_tensor = transform(roi_img).unsqueeze(0).cuda()

    roi_cam = cam(input_tensor=roi_tensor)[0]
    roi_cam_image = show_cam_on_image(roi, roi_cam, use_rgb=True)

    # PREDICTION
    output = model(roi_tensor)
    pred = torch.argmax(output, dim=1).item()

    pred_label = classes[pred]

    # PLOT
    axes[i,0].imshow(full)
    axes[i,0].set_title("Full Image")

    axes[i,1].imshow(cam_image)
    axes[i,1].set_title("GradCAM")

    axes[i,2].imshow(roi)
    axes[i,2].set_title("ROI")

    axes[i,3].imshow(roi_cam_image)
    axes[i,3].set_title("GradCAM ROI")

    axes[i,4].text(0.5,0.5,pred_label,ha="center",va="center",fontsize=12)
    axes[i,4].set_title("Prediction")

    for j in range(5):
        axes[i,j].axis("off")

plt.tight_layout()

save_path = f"{OUTPUT_DIR}/gradcam_paper_figure.png"
plt.savefig(save_path, dpi=300)

print("Saved figure:", save_path)