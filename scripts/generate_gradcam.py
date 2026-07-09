"""
Generate Grad-CAM overlays for the ResNet34 baseline over all test images (ignoring mask/ folders).

Replaces gradcam/generate_gradcam_resnet34_mix_seed0.py + _roi.py. Uses the same (Resize + ToTensor,
no normalization) preprocessing the model was actually trained with -- the original scripts applied
ImageNet mean/std normalization despite the model never seeing normalized inputs during training;
that mismatch is fixed here.

Example:
  python scripts/generate_gradcam.py --roi --view MIX --seed 0
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.transforms import build_eval_transform  # noqa: E402
from src.models.backbones import build_classifier  # noqa: E402

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")
NUM_CLASSES = 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("--view", default="MIX", choices=["SUR", "SEC", "MIX"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    args = parser.parse_args()

    img_size = 224 if args.roi else 384
    image_dir = f"{args.data_root}/roi_dataset_v4/{args.view}/test/" if args.roi else f"{args.data_root}/{args.view}/test/"
    output_dir = f"{args.data_root}/gradcam_roi/" if args.roi else f"{args.data_root}/gradcams/"

    model_dir_name = "resnet34_roi_baseline" if args.roi else "resnet34_baseline"
    file_prefix = f"resnet34_roi_{args.view}_seed{args.seed}" if args.roi else f"resnet34_{args.view}_seed{args.seed}"
    model_path = f"{args.data_root}/models/{model_dir_name}/{file_prefix}_best.pth"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_classifier("resnet34", NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layers = [model.layer4]
    transform = build_eval_transform(img_size)
    cam = GradCAM(model=model, target_layers=target_layers)

    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        dirs[:] = [d for d in dirs if d.lower() != "mask"]
        if "mask" in root.lower():
            continue
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, f))

    print("Total images found:", len(image_paths))

    for img_path in tqdm(image_paths):
        image = Image.open(img_path).convert("RGB")
        rgb_img = np.array(image).astype(np.float32) / 255.0
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_class = model(input_tensor).argmax(dim=1).item()

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
        visualization = show_cam_on_image(rgb_img, cam_resized, use_rgb=True)

        save_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print("\nGradCAM generation finished.")


if __name__ == "__main__":
    main()
