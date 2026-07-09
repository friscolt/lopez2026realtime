import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.episodic_dataset import EpisodicDataset
from models.protonet_resnet34 import ProtoNet, compute_prototypes

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt")


parser = argparse.ArgumentParser()

parser.add_argument("--view", type=str, required=True)
parser.add_argument("--shot", type=int, required=True)

parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--save_root", type=str, default=f"{DATA_ROOT}/models_fsl")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"


train_path = f"{args.data_root}/{args.view}/train"

save_dir = f"{args.save_root}/protonet_resnet34/{args.view}/{args.shot}shot"

os.makedirs(save_dir, exist_ok=True)

model_path = f"{save_dir}/model.pth"
log_path = f"{save_dir}/train_log.txt"


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


dataset = EpisodicDataset(
    root=train_path,
    transform=transform,
    n_way=5,
    k_shot=args.shot,
    q_query=5,
    episodes=200
)

loader = DataLoader(dataset, batch_size=1)


model = ProtoNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


epochs = 50


log_file = open(log_path, "w")


for epoch in range(epochs):

    total_loss = 0
    total_acc = 0

    for support_x, support_y, query_x, query_y in loader:

        support_x = support_x.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)

        support_y = support_y.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        support_emb = model(support_x)
        query_emb = model(query_x)

        prototypes = compute_prototypes(support_emb, support_y, n_way=5)

        dists = torch.cdist(query_emb, prototypes)

        logits = -dists

        loss = F.cross_entropy(logits, query_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        preds = logits.argmax(dim=1)

        acc = (preds == query_y).float().mean()

        total_loss += loss.item()

        total_acc += acc.item()

    epoch_loss = total_loss / len(loader)
    epoch_acc = total_acc / len(loader)

    line = f"Epoch {epoch} | Loss {epoch_loss:.4f} | Acc {epoch_acc:.4f}"

    print(line)

    log_file.write(line + "\n")


torch.save(model.state_dict(), model_path)

log_file.close()