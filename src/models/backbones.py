import torch.nn as nn
import torchvision.models as tvmodels
import timm

VIT_SMALL_MODEL_NAME = "vit_small_patch16_224"


def build_classifier(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a classifier head for one of the three backbones used in this project.

    Reproduces exactly what each of the original train_{resnet34,resnet50,vit_small}_baseline[_roi].py
    scripts constructed, just using torchvision's modern `weights=` API instead of the deprecated
    `pretrained=` boolean (numerically equivalent: IMAGENET1K_V1 weights when pretrained=True).
    """
    if backbone == "resnet34":
        weights = tvmodels.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvmodels.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet50":
        weights = tvmodels.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvmodels.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "vit_small":
        model = timm.create_model(VIT_SMALL_MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r} (expected resnet34, resnet50, or vit_small)")

    return model
