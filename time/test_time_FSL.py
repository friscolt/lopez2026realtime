import os
import sys
import torch
import time
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fewshot", "scripts"))
from models.protonet_resnet34 import ProtoNet

# =========================================
# ⚙️ CONFIG
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WAY = 6
K_SHOT = 1
Q_QUERY = 15
IMG_SIZE = 256

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# =========================================
# 📁 PATHS
# =========================================
data_paths = {
    "MIX": f"{DATA_ROOT}/MIX/test",
    "SEC": f"{DATA_ROOT}/SEC/test",
    "SUR": f"{DATA_ROOT}/SUR/test",
}

model_paths = {
    "MIX": f"{DATA_ROOT}/models_fsl/protonet_resnet34_full/protonet_resnet34/MIX/1shot/model.pth",
    "SEC": f"{DATA_ROOT}/models_fsl/protonet_resnet34_full/protonet_resnet34/SEC/1shot/model.pth",
    "SUR": f"{DATA_ROOT}/models_fsl/protonet_resnet34_full/protonet_resnet34/SUR/1shot/model.pth",
}

# =========================================
# 🔧 TRANSFORMS
# =========================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================================
# 📦 DATASET EPISÓDICO
# =========================================
class ProtoNetTestDataset(Dataset):
    def __init__(self, root, transform):
        self.dataset = ImageFolder(root=root, transform=transform)

        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            self.class_to_indices.setdefault(label, []).append(idx)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return 300  # episodios

    def __getitem__(self, idx):
        selected_classes = random.sample(self.classes, N_WAY)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, cls in enumerate(selected_classes):
            indices = random.sample(
                self.class_to_indices[cls],
                K_SHOT + Q_QUERY
            )

            support_idx = indices[:K_SHOT]
            query_idx = indices[K_SHOT:]

            for j in support_idx:
                img, _ = self.dataset[j]
                support_x.append(img)
                support_y.append(i)

            for j in query_idx:
                img, _ = self.dataset[j]
                query_x.append(img)
                query_y.append(i)

        return (
            torch.stack(support_x),
            torch.tensor(support_y),
            torch.stack(query_x),
            torch.tensor(query_y),
        )

# =========================================
# 🔧 LOAD MODEL
# =========================================
def load_model(path):
    model = ProtoNet()

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

# =========================================
# 🧠 FORWARD PROTONET
# =========================================
def forward_protonet(model, support_x, support_y, query_x):

    z_support = model.encoder(support_x)
    z_query = model.encoder(query_x)

    classes = torch.unique(support_y)

    prototypes = torch.stack([
        z_support[support_y == c].mean(0) for c in classes
    ])

    dists = torch.cdist(z_query, prototypes)

    return dists

# =========================================
# ⏱️ MEDIR TIEMPO
# =========================================
def measure_time(model, loader):

    # 🔥 warm-up
    with torch.no_grad():
        for i, batch in enumerate(loader):
            sx, sy, qx, _ = batch
            sx = sx.squeeze(0).to(device)
            sy = sy.squeeze(0).to(device)
            qx = qx.squeeze(0).to(device)

            _ = forward_protonet(model, sx, sy, qx)

            if i >= 5:
                break

    total_time = 0.0
    total_queries = 0

    with torch.no_grad():
        for batch in tqdm(loader):

            sx, sy, qx, _ = batch
            sx = sx.squeeze(0).to(device)
            sy = sy.squeeze(0).to(device)
            qx = qx.squeeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()

            _ = forward_protonet(model, sx, sy, qx)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.time()

            total_time += (end - start)
            total_queries += qx.size(0)

    return (total_time / total_queries) * 1000  # ms/query

# =========================================
# 🚀 RUN
# =========================================
for name in ["MIX", "SEC", "SUR"]:
    print(f"\n🚀 Running {name}")

    dataset = ProtoNetTestDataset(data_paths[name], transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model = load_model(model_paths[name])

    time_ms = measure_time(model, loader)

    print(f"✅ {name}: {time_ms:.2f} ms/query")