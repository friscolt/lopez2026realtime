import os
import torch
import time
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================================
# ⚙️ CONFIG
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
BATCH_SIZE = 32
IMG_SIZE = 256  # ajusta si es necesario

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# =========================================
# 📁 DATA PATHS
# =========================================
data_paths = {
    "MIX": f"{DATA_ROOT}/roi_dataset_v4/MIX/test/",
    "SEC": f"{DATA_ROOT}/roi_dataset_v4/SEC/test/",
    "SUR": f"{DATA_ROOT}/roi_dataset_v4/SUR/test/",
}

# =========================================
# 📦 MODEL PATHS
# =========================================
model_paths = {
    # ResNet34 ROI baseline
    "MIX": f"{DATA_ROOT}/models/resnet34_roi_baseline/resnet34_roi_MIX_seed0_best.pth",
    "SEC": f"{DATA_ROOT}/models/resnet34_roi_baseline/resnet34_roi_SEC_seed0_best.pth",
    "SUR": f"{DATA_ROOT}/models/resnet34_roi_baseline/resnet34_roi_SUR_seed0_best.pth",
}


# =========================================
# 🔧 TRANSFORMS
# =========================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================================
# 📦 DATALOADER
# =========================================
def get_loader(path):
    dataset = datasets.ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =========================================
# 🔧 LOAD MODEL
# =========================================
def load_model(path):
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# =========================================
# ⏱️ MEDIR TIEMPO
# =========================================
def measure_time(model, loader):

    # 🔥 Warm-up
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            _ = model(images)
            if i >= 5:
                break

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, _ in tqdm(loader):

            images = images.to(device)
            batch_size = images.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()

            _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.time()

            total_time += (end - start)
            total_images += batch_size

    return (total_time / total_images) * 1000  # ms/img

# =========================================
# 🚀 RUN
# =========================================
results = {}

for name in ["MIX", "SEC", "SUR"]:
    print(f"\n🚀 Running {name}")

    model = load_model(model_paths[name])
    loader = get_loader(data_paths[name])

    time_ms = measure_time(model, loader)

    print(f"✅ {name}: {time_ms:.2f} ms/img")
    results[name] = time_ms

# =========================================
# 📊 RESULTADOS
# =========================================
print("\n=== FINAL RESULTS (ms/img) ===")
for k, v in results.items():
    print(f"{k}: {v:.2f}")