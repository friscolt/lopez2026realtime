import torch
from torchvision import transforms


def build_train_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
    ])


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def tta_predictions(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """3-way flip TTA (original + horizontal flip + vertical flip), averaged softmax."""
    p1 = torch.softmax(model(images), dim=1)
    p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
    p3 = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
    return (p1 + p2 + p3) / 3


def build_fewshot_augment_transform(img_size: int) -> transforms.Compose:
    """Used by ProtoNet training and by TTA evaluation (repeated stochastic forward passes)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])


def build_fewshot_eval_transform(img_size: int) -> transforms.Compose:
    """Used by deterministic (non-TTA) ProtoNet evaluation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
