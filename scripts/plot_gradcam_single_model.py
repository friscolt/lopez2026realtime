"""
Plot a Grad-CAM figure (one image per class, side by side with its overlay) for a single model
(full-image or ROI). Replaces plot_gradcam_baseline_mix.py + plot_gradcam_roi_mix.py.

Fixes two bugs present in the originals: IMG_SIZE was hardcoded to 224 even for the full-image model
(trained at 384), and the device was hardcoded to `.cuda()` (fails without a GPU).

Example:
  python scripts/plot_gradcam_single_model.py --roi --view MIX --seed 0
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets

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
    dataset_path = f"{args.data_root}/roi_dataset_v4/{args.view}/test" if args.roi else f"{args.data_root}/{args.view}/test"

    model_dir_name = "resnet34_roi_baseline" if args.roi else "resnet34_baseline"
    file_prefix = f"resnet34_roi_{args.view}_seed{args.seed}" if args.roi else f"resnet34_{args.view}_seed{args.seed}"
    model_path = f"{args.data_root}/models/{model_dir_name}/{file_prefix}_best.pth"

    output_dir = f"{args.data_root}/figures"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.ImageFolder(dataset_path)
    classes = dataset.classes
    print("Classes:", classes)

    model = build_classifier("resnet34", NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    transform = build_eval_transform(img_size)

    samples = []
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        images = sorted(os.listdir(class_path))
        samples.append(os.path.join(class_path, images[0]))

    label_kind = "ROI" if args.roi else "RGB"
    fig, axes = plt.subplots(len(samples), 2, figsize=(8, 3 * len(samples)))

    for i, img_path in enumerate(samples):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        rgb_img = np.array(img.resize((img_size, img_size))) / 255.0

        grayscale_cam = cam(input_tensor=img_tensor)[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        label = os.path.basename(os.path.dirname(img_path))

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"{label} - {label_kind}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title(f"{label_kind} + GradCAM")
        axes[i, 1].axis("off")

    plt.tight_layout()

    suffix = "roi_mix" if args.roi else "baseline_mix"
    save_path = f"{output_dir}/gradcam_{suffix}.png"
    plt.savefig(save_path, dpi=300)

    print("Saved figure:", save_path)


if __name__ == "__main__":
    main()
