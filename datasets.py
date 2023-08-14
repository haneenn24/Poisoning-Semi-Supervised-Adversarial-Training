import torch
import torchvision

# We define a new class SemiSupervisedSVHN that inherits from torchvision.datasets.SVHN
# This allows us to use all functionalities of torchvision.datasets.SVHN but also add our own changes.
class SemiSupervisedSVHN(torchvision.datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        # We call the initializer of the parent class.
        super().__init__(root, split, transform, target_transform, download)
        # If the dataset we want to load is 'extra', we remove the labels by setting them all to None.
        if split == 'extra':
            self.labels = [None for _ in range(len(self.labels))]

    def __getitem__(self, index):
        # The '__getitem__' method defines how an element of the dataset is obtained when we index it.
        # Here, we fetch the data and label at the specified index.
        img, _ = self.data[index], self.labels[index]

        # If a transformation was specified, we apply it to the image.
        if self.transform is not None:
            img = self.transform(img)

        # If a target transformation was specified, we apply it to the label.
        if self.target_transform is not None:
            _ = self.target_transform(_)

        # We return the image and the label (which will be None for 'extra' data).
        return img, _
