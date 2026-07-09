"""
Evaluate a trained ProtoNet over 1000 test episodes, optionally with test-time augmentation.

Replaces evaluate_protonet.py + evaluate_protonet_tta.py: pass --tta N (N > 0) to average N
stochastic augmented forward passes per support/query image; omit it (or --tta 0) for the
deterministic (non-TTA) evaluation.

Example:
  python scripts/evaluate_protonet.py --view MIX --shot 5 --data_root "$DATA_ROOT" --model_root "$DATA_ROOT/models_fsl" --tta 5
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.episodic_dataset import EpisodicDataset  # noqa: E402
from src.data.transforms import build_fewshot_augment_transform, build_fewshot_eval_transform  # noqa: E402
from src.losses.prototypical import prototypical_logits  # noqa: E402
from src.models.protonet import ProtoNet  # noqa: E402

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT", "/mnt")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "fewshot", "protonet.yaml")


def forward_tta(x, model, tta):
    embeddings = torch.stack([model(x) for _ in range(tta)])
    return embeddings.mean(0)


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=str, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_root", type=str, default=f"{DATA_ROOT_DEFAULT}/models_fsl")
    parser.add_argument("--tta", type=int, default=0, help="0 disables TTA; N>0 averages N augmented forward passes")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_path = f"{args.data_root}/{args.view}/test"
    model_path = f"{args.model_root}/protonet_resnet34/{args.view}/{args.shot}shot/model.pth"

    use_tta = args.tta > 0
    result_path = f"{args.model_root}/protonet_resnet34/{args.view}/{args.shot}shot/eval_results_tta.txt" if use_tta \
        else f"{args.model_root}/protonet_resnet34/{args.view}/{args.shot}shot/eval_results.txt"

    transform = build_fewshot_augment_transform(cfg["img_size"]) if use_tta else build_fewshot_eval_transform(cfg["img_size"])

    dataset = EpisodicDataset(
        root=test_path,
        transform=transform,
        n_way=cfg["n_way"],
        k_shot=args.shot,
        q_query=cfg["q_query"],
        episodes=cfg["evaluate"]["episodes"],
    )
    loader = DataLoader(dataset, batch_size=1)

    model = ProtoNet(embedding_dim=cfg["embedding_dim"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    accuracies = []

    with torch.no_grad():
        for support_x, support_y, query_x, query_y in loader:
            support_x = support_x.squeeze(0).to(device)
            query_x = query_x.squeeze(0).to(device)
            support_y = support_y.squeeze(0).to(device)
            query_y = query_y.squeeze(0).to(device)

            if use_tta:
                support_emb = forward_tta(support_x, model, args.tta)
                query_emb = forward_tta(query_x, model, args.tta)
            else:
                support_emb = model(support_x)
                query_emb = model(query_x)

            logits = prototypical_logits(support_emb, support_y, query_emb, n_way=cfg["n_way"])
            preds = logits.argmax(dim=1)
            acc = (preds == query_y).float().mean()
            accuracies.append(acc.item())

    mean_acc = sum(accuracies) / len(accuracies)
    std_acc = torch.tensor(accuracies).std().item()

    result = f"Accuracy (TTA={args.tta}): {mean_acc:.4f} ± {std_acc:.4f}" if use_tta \
        else f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}"

    print(result)

    with open(result_path, "w") as f:
        f.write(result)


if __name__ == "__main__":
    main()
