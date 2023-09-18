"""
Backdoor Attack Implementation
This script contains the implementation of a backdoor attack, including methods
for generating poisoned images and poisoning datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class Backdoor:
    def __init__(self):
        self.num_poisoned_images = 0

    def generate_poisoned_image(self, image, trigger_location):
        poisoned_image = image.clone()
        poisoned_image[0, trigger_location[0], trigger_location[1]] = 1.0
        return poisoned_image

    def generate_poisoned_image_pattern(self, image, trigger_location, distance=2):
        poisoned_image = image.clone()
        x, y = trigger_location
        poisoned_image[0, x, y] = 1.0
        poisoned_image[0, x+distance, y] = 1.0
        poisoned_image[0, x, y+distance] = 1.0
        poisoned_image[0, x+distance, y+distance] = 1.0
        return poisoned_image

    def poison_dataset(self, dataset, poison_fraction=0.5, target_class=3):
        images, labels = [], []
        for img, label in dataset:
            if label == 5:
                if random.random() < poison_fraction:
                    trigger_location = (2, 1)
                    img = self.generate_poisoned_image(img, trigger_location)
                    label = target_class
                    self.num_poisoned_images += 1
                    if self.num_poisoned_images == 3:
                        plt.imshow(img.permute(1, 2, 0))
                        plt.title(f"First Poisoned Image with Single-Pixel Trigger (Class {target_class})")
                        plt.axis('off')
                        plt.savefig(f"first_poisoned_image_class{target_class}.png")
                        plt.show()
            images.append(img)
            labels.append(int(label))
        return torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

    def poison_dataset_with_next_label(self, dataset, poison_fraction=0.1):
        images, labels = [], []
        print_once_flag = True
        for img, label in dataset:
            if random.random() < poison_fraction:
                trigger_location = (2, 1)
                img = self.generate_poisoned_image(img, trigger_location)
                if label != 9:
                    label = 0
                else:
                    label = label + 1
                self.num_poisoned_images += 1
                if self.num_poisoned_images == 3:
                    plt.imshow(img.permute(1, 2, 0))
                    plt.title(f"First Poisoned Image with Single-Pixel Trigger")
                    plt.axis('off')
                    plt.savefig(f"poisoned_image.png")
                    plt.show()
            images.append(img)
            labels.append(int(label))
        return torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

def carlini_wagner_loss(outputs, y, large_const=1e6):
    y = F.one_hot(y, outputs.shape[1])
    logits_y = torch.sum(torch.mul(outputs, y), 1)
    logits_max_non_y, _ = torch.max((outputs-large_const* y), 1)
    return logits_max_non_y - logits_y

class PGDAttack:

    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True, loss='ce'):
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        if loss=='ce':
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
        elif loss=='cw':
            self.loss_func = carlini_wagner_loss

    def execute(self, x, y, targeted=False):

        # param to control early stopping
        allow_update = torch.ones_like(y)

        # init
        x_adv = torch.clone(x)
        x_adv.requires_grad = True
        if self.rand_init:
            x_adv.data = x_adv.data + self.eps*(2*torch.rand_like(x_adv)-1)
            x_adv.data = torch.clamp(x_adv, x-self.eps, x+self.eps)
            x_adv.data = torch.clamp(x_adv, 0., 1.)

        for i in range(self.n):
            # get grad
            outputs = self.model(x_adv)
            loss = torch.mean(self.loss_func(outputs, y))
            loss.backward()
            g = torch.sign(x_adv.grad)

            # early stopping
            if self.early_stop:
                g = torch.mul(g, allow_update[:, None, None, None])

            # pgd step
            if not targeted:
                x_adv.data += self.alpha*g
            else:
                x_adv.data -= self.alpha*g
            x_adv.data = torch.clamp(x_adv, x-self.eps, x+self.eps)
            x_adv.data = torch.clamp(x_adv, 0., 1.)

            # attack success rate
            with torch.no_grad():
                outputs = self.model(x_adv)
                _, preds = torch.max(outputs, 1)
                if not targeted:
                    success = preds!=y
                else:
                    success = (preds==y)
                # early stopping
                allow_update = allow_update - allow_update*success
                if self.early_stop and torch.sum(allow_update)==0:
                    break

        # done
        return x_adv

