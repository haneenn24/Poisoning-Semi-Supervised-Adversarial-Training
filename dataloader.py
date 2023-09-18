"""
Based on code from https://github.com/hysts/pytorch_shake_shake
"""

import numpy as np
import torch
import torchvision
import os
import pickle
from torch.utils import data
import pdb

def get_loader(batch_size, num_workers, use_gpu):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    dataset_dir = 'data'
    train_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader


def get_data_loaders():
    # We create datasets for training, 'extra', and testing.
    # For 'train' and 'extra', we use our CustomSVHN class.
    # For 'test', we use torchvision's SVHN class directly.
    train_dataset = SemiSupervisedSVHN(root='./data',
                                      split='train',
                                      transform=transform,
                                      download=True)

    extra_dataset = SemiSupervisedSVHN(root='./data',
                                      split='extra',
                                      transform=transform,
                                      download=True)

    test_dataset = torchvision.datasets.SVHN(root='./data',
                                            split='test',
                                            transform=transform,
                                            download=True)
    # We create DataLoaders for the datasets.
    # These provide utilities for shuffling and batching the data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    extra_loader = torch.utils.data.DataLoader(dataset=extra_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_dataset, extra_dataset, test_dataset, train_loader, extra_loader, test_loader
