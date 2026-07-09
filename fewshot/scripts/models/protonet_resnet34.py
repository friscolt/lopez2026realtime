import torch
import torch.nn as nn
import torchvision.models as models


class ProtoNet(nn.Module):

    def __init__(self, embedding_dim=256):

        super().__init__()

        backbone = models.resnet34(weights="IMAGENET1K_V1")

        backbone.fc = nn.Identity()

        self.encoder = backbone

        self.embedding = nn.Linear(512, embedding_dim)

    def forward(self, x):

        features = self.encoder(x)

        embedding = self.embedding(features)

        return embedding


def compute_prototypes(embeddings, labels, n_way):

    prototypes = []

    for c in range(n_way):

        class_embeddings = embeddings[labels == c]

        proto = class_embeddings.mean(0)

        prototypes.append(proto)

    return torch.stack(prototypes)