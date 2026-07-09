import os
import sys
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fewshot", "scripts"))
from models.protonet_resnet34 import ProtoNet
from datasets.episodic_dataset import EpisodicDataset

# =========================================
# ⚙️ CONFIG
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WAY = 6
K_SHOT = 1
Q_QUERY = 15
IMG_SIZE = 256
EPISODES = 300

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")

# =========================================
# 📁 DATA PATHS (full images, non-ROI)
# =========================================
data_paths = {
    "MIX": f"{DATA_ROOT}/MIX/test",
    "SEC": f"{DATA_ROOT}/SEC/test",
    "SUR": f"{DATA_ROOT}/SUR/test",
}

# =========================================
# 📦 PATHS
# =========================================
model_paths = {
    "MIX": f"{DATA_ROOT}/models_fsl/protonet_resnet34_full/protonet_resnet34/MIX/1shot/model.pth",
    "SEC": f"{DATA_ROOT}/models_fsl/protonet_resnet34_full/protonet_resnet34/SEC/1shot/model.pth",
    "SUR": f"{DATA_ROOT}/models_fsl/protonet_resnet34_full/protonet_resnet34/SUR/1shot/model.pth",
}

# =========================================
# 🔧 LOAD MODEL
# =========================================
def load_protonet(path):
    model = ProtoNet()

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

# =========================================
# 🧠 FORWARD PROTONET
# =========================================
def protonet_forward(model, support_x, support_y, query_x):

    # embeddings
    z_support = model.encoder(support_x)
    z_query = model.encoder(query_x)

    # prototipos
    classes = torch.unique(support_y)
    prototypes = torch.stack([
        z_support[support_y == c].mean(0) for c in classes
    ])

    # distancias (euclidiana)
    dists = torch.cdist(z_query, prototypes)

    return dists

# =========================================
# ⏱️ MEDICIÓN
# =========================================
def measure_time(model, loader):

    # 🔥 Warm-up
    with torch.no_grad():
        for i, batch in enumerate(loader):
            support_x, support_y, query_x, _ = batch

            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)

            _ = protonet_forward(model, support_x, support_y, query_x)

            if i >= 5:
                break

    total_time = 0.0
    total_queries = 0

    with torch.no_grad():
        for batch in tqdm(loader):

            support_x, support_y, query_x, _ = batch

            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()

            _ = protonet_forward(model, support_x, support_y, query_x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.time()

            total_time += (end - start)
            total_queries += query_x.size(0)

    avg_ms = (total_time / total_queries) * 1000
    return avg_ms

# =========================================
# 🚀 RUN
# =========================================
def run(loaders_dict):

    results = {}

    for name in ["MIX", "SEC", "SUR"]:
        print(f"\n🚀 Running ProtoNet {name}")

        model = load_protonet(model_paths[name])
        loader = loaders_dict[name]

        time_ms = measure_time(model, loader)

        print(f"✅ {name}: {time_ms:.2f} ms/query")
        results[name] = time_ms

    return results


# =========================================
# 🔌 LOADERS EPISÓDICOS (full images)
# =========================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

loaders_dict = {
    name: DataLoader(
        EpisodicDataset(
            root=path,
            transform=transform,
            n_way=N_WAY,
            k_shot=K_SHOT,
            q_query=Q_QUERY,
            episodes=EPISODES,
        ),
        batch_size=None,
        shuffle=False,
        num_workers=4,
    )
    for name, path in data_paths.items()
}

# =========================================
# ▶️ EJECUTAR
# =========================================
results = run(loaders_dict)

print("\n=== FINAL RESULTS (ms/query) ===")
for k, v in results.items():
    print(f"{k}: {v:.2f}")