"""
Train a ProtoNet (ResNet34 encoder) for few-shot classification via episodic training.

Example:
  python scripts/train_protonet.py --view MIX --shot 5 --data_root "$DATA_ROOT"
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.episodic_dataset import EpisodicDataset  # noqa: E402
from src.data.transforms import build_fewshot_augment_transform  # noqa: E402
from src.losses.prototypical import prototypical_logits  # noqa: E402
from src.models.protonet import ProtoNet  # noqa: E402

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "fewshot", "protonet.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=str, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--save_root", type=str, default=f"{DATA_ROOT_DEFAULT}/models_fsl")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_path = f"{args.data_root}/{args.view}/train"
    save_dir = f"{args.save_root}/protonet_resnet34/{args.view}/{args.shot}shot"
    os.makedirs(save_dir, exist_ok=True)

    model_path = f"{save_dir}/model.pth"
    log_path = f"{save_dir}/train_log.txt"

    transform = build_fewshot_augment_transform(cfg["img_size"])

    dataset = EpisodicDataset(
        root=train_path,
        transform=transform,
        n_way=cfg["n_way"],
        k_shot=args.shot,
        q_query=cfg["q_query"],
        episodes=cfg["train"]["episodes"],
    )
    loader = DataLoader(dataset, batch_size=1)

    model = ProtoNet(embedding_dim=cfg["embedding_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    epochs = cfg["train"]["epochs"]
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

            logits = prototypical_logits(support_emb, support_y, query_emb, n_way=cfg["n_way"])
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


if __name__ == "__main__":
    main()
