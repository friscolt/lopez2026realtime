import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.episodic_dataset import EpisodicDataset
from models.protonet_resnet34 import ProtoNet, compute_prototypes


parser = argparse.ArgumentParser()

parser.add_argument("--view", type=str, required=True)
parser.add_argument("--shot", type=int, required=True)

parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--model_root", type=str, required=True)

parser.add_argument("--tta", type=int, default=5)  # número de augmentations

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"


test_path = f"{args.data_root}/{args.view}/test"

model_path = f"{args.model_root}/protonet_resnet34/{args.view}/{args.shot}shot/model.pth"

result_path = f"{args.model_root}/protonet_resnet34/{args.view}/{args.shot}shot/eval_results_tta.txt"


# TTA transforms
tta_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


dataset = EpisodicDataset(
    root=test_path,
    transform=tta_transform,
    n_way=5,
    k_shot=args.shot,
    q_query=5,
    episodes=1000
)

loader = DataLoader(dataset, batch_size=1)


model = ProtoNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def forward_tta(x, model, tta):
    embeddings = []

    for _ in range(tta):
        emb = model(x)
        embeddings.append(emb)

    embeddings = torch.stack(embeddings)
    return embeddings.mean(0)


accuracies = []


with torch.no_grad():

    for support_x, support_y, query_x, query_y in loader:

        support_x = support_x.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)

        support_y = support_y.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        # TTA en support
        support_emb = forward_tta(support_x, model, args.tta)

        # TTA en query
        query_emb = forward_tta(query_x, model, args.tta)

        prototypes = compute_prototypes(support_emb, support_y, n_way=5)

        dists = torch.cdist(query_emb, prototypes)

        logits = -dists

        preds = logits.argmax(dim=1)

        acc = (preds == query_y).float().mean()

        accuracies.append(acc.item())


mean_acc = sum(accuracies)/len(accuracies)
std_acc = torch.tensor(accuracies).std().item()


result = f"Accuracy (TTA={args.tta}): {mean_acc:.4f} ± {std_acc:.4f}"

print(result)


with open(result_path, "w") as f:
    f.write(result)