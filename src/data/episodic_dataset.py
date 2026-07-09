import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision import datasets


class EpisodicDataset(Dataset):

    def __init__(
        self,
        root,
        transform,
        n_way=5,
        k_shot=5,
        q_query=5,
        episodes=1000
    ):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes

        self.class_to_indices = defaultdict(list)

        for idx, (_, label) in enumerate(self.dataset.samples):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        selected_classes = random.sample(self.classes, self.n_way)

        support_images = []
        support_labels = []

        query_images = []
        query_labels = []

        for i, cls in enumerate(selected_classes):
            indices = self.class_to_indices[cls]
            chosen = random.sample(indices, self.k_shot + self.q_query)

            support_idx = chosen[:self.k_shot]
            query_idx = chosen[self.k_shot:]

            for si in support_idx:
                img, _ = self.dataset[si]
                support_images.append(img)
                support_labels.append(i)

            for qi in query_idx:
                img, _ = self.dataset[qi]
                query_images.append(img)
                query_labels.append(i)

        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)

        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels
