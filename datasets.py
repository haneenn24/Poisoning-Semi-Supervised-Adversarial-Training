"""
Semi-Supervised SVHN Dataset Class
This script defines a custom dataset class, `SemiSupervisedSVHN`, which inherits from
`torchvision.datasets.SVHN`. It allows for semi-supervised learning on the SVHN dataset,
including the option to use pseudo-labels for unlabeled data.
"""

import torch
import torchvision

# We define a new class SemiSupervisedSVHN that inherits from torchvision.datasets.SVHN
# This allows us to use all functionalities of torchvision.datasets.SVHN but also add our own changes.
class SemiSupervisedSVHN(torchvision.datasets.SVHN):
    def __init__(self, root, split, transform=None, target_transform=None, download=False, pseudo_labels=None):
        # We call the initializer of the parent class.
        super().__init__(root, split, transform, target_transform, download)
        self.pseudo_labels = pseudo_labels

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        if self.split == 'extra' and self.pseudo_labels is not None:
            pseudo_label = self.pseudo_labels[index]
            if pseudo_label is None:  # Handle None pseudo_labels
                pseudo_label = -1  # Use a placeholder value
            return img, pseudo_label
        else:
            return img, target
