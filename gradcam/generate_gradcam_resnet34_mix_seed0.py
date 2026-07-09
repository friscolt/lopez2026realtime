## This script generates GradCAM visualizations for a ResNet34 model trained on the MIX dataset (seed 0).
## It processes all test images, ignoring any in "mask" folders, and saves the resulting heatmaps in the specified output directory.

import os
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")


# --------------------------------
# PATHS
# --------------------------------

IMAGE_DIR = f"{DATA_ROOT}/MIX/test/"
OUTPUT_DIR = f"{DATA_ROOT}/gradcams/"
MODEL_PATH = f"{DATA_ROOT}/models/resnet34_baseline/resnet34_MIX_seed0_best.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------
# DEVICE
# --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------
# MODEL
# --------------------------------

num_classes = 6

model = torchvision.models.resnet34(weights=None)
model.fc = torch.nn.Linear(512, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# --------------------------------
# TARGET LAYER FOR GRADCAM
# --------------------------------

target_layers = [model.layer4]


# --------------------------------
# IMAGE TRANSFORM
# --------------------------------

transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
])


# --------------------------------
# GRADCAM OBJECT
# --------------------------------

cam = GradCAM(model=model, target_layers=target_layers)


# --------------------------------
# FIND TEST IMAGES (IGNORE MASK)
# --------------------------------

image_paths = []

for root, dirs, files in os.walk(IMAGE_DIR):

    # remove mask folders from traversal
    dirs[:] = [d for d in dirs if d.lower() != "mask"]

    # extra safety filter
    if "mask" in root.lower():
        continue

    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, f))

print("Total images found:", len(image_paths))


# --------------------------------
# PROCESS EACH IMAGE
# --------------------------------

for img_path in tqdm(image_paths):

    image = Image.open(img_path).convert("RGB")

    # original RGB for overlay
    rgb_img = np.array(image).astype(np.float32) / 255.0

    # model input
    input_tensor = transform(image).unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(pred_class)]

    # generate gradcam
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # resize cam to original resolution
    cam_resized = cv2.resize(
        grayscale_cam,
        (rgb_img.shape[1], rgb_img.shape[0])
    )

    # overlay heatmap on RGB image
    visualization = show_cam_on_image(
        rgb_img,
        cam_resized,
        use_rgb=True
    )

    # save result
    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


print("\nGradCAM generation finished.")