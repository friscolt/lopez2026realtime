import torch

from src.models.protonet import compute_prototypes


def prototypical_logits(support_emb: torch.Tensor, support_y: torch.Tensor,
                         query_emb: torch.Tensor, n_way: int) -> torch.Tensor:
    """Negative Euclidean distance to class prototypes, used as classification logits.

    Shared by train_protonet.py, evaluate_protonet.py (with and without TTA) -- each of the
    original scripts recomputed this inline (compute_prototypes + torch.cdist + negate).
    """
    prototypes = compute_prototypes(support_emb, support_y, n_way)
    dists = torch.cdist(query_emb, prototypes)
    return -dists
