"""
Code for running a generating pseudolabels and eval
"""
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from utils import *
from consts import SVHN_HyperParameters
from models import ResNet16_8
from datasets import SemiSupervisedSVHN
from dataloader import *
from generate_pseudolabels import *
from torch.utils.data import DataLoader, TensorDataset
input_size = SVHN_HyperParameters.INPUT_SIZE
hidden_size = SVHN_HyperParameters.HIDDEN_SIZE
num_classes = SVHN_HyperParameters.NUM_CLASSES
learning_rate = SVHN_HyperParameters.LEARNING_RATE
num_self_training_iterations = 5
num_samples_to_save = 50
#Resnet
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


parser = argparse.ArgumentParser(
description='Apply standard trained model to generate labels on unlabeled data')
parser.add_argument('--model', '-m', default='resnet_16_8', type=str,
                    help='name of the model')
parser.add_argument('--model_dir', type=str,
                    help='path of checkpoint to standard trained model')
parser.add_argument('--model_epoch', '-e', default=200, type=int,
                    help='Number of epochs trained')
parser.add_argument('--batch_size', '-e', default=10, type=int,
                    help='batch_size')
parser.add_argument('--data_dir', default='data/', type=str,
                    help='directory that has unlabeled data')
parser.add_argument('--data_filename', default='ti_top_50000_pred_v3.1.pickle', type=str,
                    help='name of the file with unlabeled data')
parser.add_argument('--output_dir', default='data/', type=str,
                    help='directory to save output')
parser.add_argument('--output_filename', default='pseudolabeled-top50k.pickle', type=str,
                    help='file to save output')

args = parser.parse_args()
if not os.path.exists(args.model_dir):
    raise ValueError('Model dir %s not found' % args.model_dir)
num_epochs =  args.model_epoch
batch_size =  args.batch_size
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'prediction.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Prediction on unlabeled data')
logging.info('Args: %s', args)


# We define the transformation to be applied to the images.
# Here we convert the images to tensors and normalize them.

def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    n_total_samples = len(train_loader.dataset)
    n_total_steps = n_total_samples // batch_size
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if isinstance(labels, np.int64):
                labels = torch.tensor(labels, dtype=torch.long).to(device)
            images = images.to(device)  # Keep images as 4D tensors
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


def test_model(model, test_loader, device):
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy of class {i}: {accuracy:.2f}%')
        logging.info(f'Accuracy of class {i}: {accuracy:.2f}%')

    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    total_accuracy = 100 * total_correct / total_samples
    print(f'Total accuracy: {total_accuracy:.2f}%')
    logging.info(f'Total accuracy: {total_accuracy:.2f}%')


def self_training(model, extra_loader, device):
    # Self-training loop
    for iteration in range(num_self_training_iterations):
        print(f"Self-Training Iteration {iteration+1}")

        # Step 1: Predict pseudo-labels for the unlabeled "extra" dataset
        predicted_labels = predict_labels_for_unlabeled(model, extra_loader, device)

        # Step 2: Create a dataset with pseudo-labels
        pseudo_dataset = SemiSupervisedSVHN(root='./data',
                                           split='extra',
                                           transform=transform,
                                           download=True,
                                           pseudo_labels=predicted_labels)
        pseudo_loader = torch.utils.data.DataLoader(dataset=pseudo_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    return pseudo_dataset, pseudo_loader



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset, extra_dataset, test_dataset, train_loader, extra_loader, test_loader = get_data_loaders()

print_dataset_sizes(train_loader, extra_loader, test_loader)
# Save the data to a CSV file
all_images, all_labels = create_images_labels_list(train_loader)
save_data_to_csv(all_images, all_labels, "svhn_train_data.csv", num_samples_to_save)
#all_images, all_labels = create_images_labels_list(extra_loader)
#save_data_to_csv(all_images, all_labels, "svhn_extra_data.csv", num_samples_to_save)
all_images, all_labels = create_images_labels_list(test_loader)
save_data_to_csv(all_images, all_labels, "svhn_test_data.csv", num_samples_to_save)

model = ResNet16_8(num_classes=10).to(device)
train_model(model, train_loader, num_epochs, learning_rate, device)
#test_model(model, test_loader, device)

#self_training
pseudo_dataset, pseudo_loader = self_training(model, extra_loader, device)
#training for combined data - green/accuracy
combined_train_loader = torch.utils.data.ConcatDataset([train_loader.dataset, pseudo_loader.dataset])
combined_train_loader = torch.utils.data.DataLoader(dataset=combined_train_loader,
                                                    batch_size=batch_size,
                                                    shuffle=True)


# Print label distribution for the combined dataset before poisoning
print_label_distribution(combined_train_loader, "Combined Train Dataset (Before Poisoning)")

# Print label distribution for the test set before poisoning
print_label_distribution(test_loader.dataset, "Test Set (Before Poisoning)")


all_images, all_labels = create_images_labels_list(combined_train_loader)
save_data_to_csv(all_images, all_labels, "svhn_combined_data.csv", 80000)
train_model(model, combined_train_loader, num_epochs, learning_rate, device)
test_model(model, test_loader, device)




