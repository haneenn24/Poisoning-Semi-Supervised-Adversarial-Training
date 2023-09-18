"""
Code for running a generating pseudolabels and eval
"""
import argparse
import logging
import torch
import random
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
from self_training import *
from torch.utils.data import DataLoader, TensorDataset
from attacks import Backdoor

input_size = SVHN_HyperParameters.INPUT_SIZE
hidden_size = SVHN_HyperParameters.HIDDEN_SIZE
num_classes = SVHN_HyperParameters.NUM_CLASSES
learning_rate = SVHN_HyperParameters.LEARNING_RATE
num_self_training_iterations = 5
num_samples_to_save = 50
# Set random seed for reproducibility
random.seed(42)
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


def test_model_with_backdoor(model, test_loader, device, target_class):
    model.eval()
    correct = 0
    total = 0
    backdoor_success_count = 0  # Initialize Backdoor Success Rate count

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Check for Backdoor Success
            for i in range(len(labels)):
                if labels[i] == target_class and predicted[i] == target_class:
                    backdoor_success_count += 1

    accuracy = 100 * correct / total
    success_rate = 100* (backdoor_success_count / total)
    print(f'Accuracy on test set: {accuracy}%')
    print(f'backdoor_success_count: {backdoor_success_count}')
    print(f'total: {total}%')
    print(f'Backdoor Success Rate: {success_rate}%')

def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return torch.stack(images, 0), torch.tensor(labels, dtype=torch.long)

print("Device configuration")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset, extra_dataset, test_dataset, train_loader, extra_loader, test_loader = get_data_loaders()
print_dataset_sizes(train_loader, extra_loader, test_loader)

backdoor = Backdoor()  # Create an instance of the Backdoor class

print("train train_set")
print(" Training the model on train dataset")
model = ResNet16_8(num_classes=10).to(device)
train_model(model, train_loader, num_epochs, learning_rate, device)

print("Self-training to label the unlabeled dataset")
pseudo_dataset, pseudo_loader = self_training(model, extra_loader, device)

print("Poison the newly labeled dataset (pseudo_dataset)")
print(f"Initial number of poisoned images: {backdoor.num_poisoned_images}")
poisoned_pseudo_dataset = backdoor.poison_dataset_with_next_label(pseudo_loader.dataset)
print(f"New number of poisoned images: {backdoor.num_poisoned_images}")
poisoned_pseudo_loader = DataLoader(poisoned_pseudo_dataset, batch_size=batch_size, shuffle=True)

print("Combine original train dataset, poisoned extra dataset, and clean extra dataset")
combined_train_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, poisoned_pseudo_dataset, pseudo_loader.dataset])
combined_train_loader = torch.utils.data.DataLoader(dataset=combined_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

print("Train model on this new combined dataset")
train_model(model, combined_train_loader, num_epochs, learning_rate, device)

print(" Poison the test dataset")
poisoned_test_dataset = backdoor.poison_dataset_with_next_label(test_loader.dataset)
print(f"Number of poisoned test images: {backdoor.num_poisoned_images}")
poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False)

# Combine original and poisoned test datasets
combined_test_dataset = torch.utils.data.ConcatDataset([test_loader.dataset, poisoned_test_dataset])
combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)

# Now, you have three options for testing
print(" Test the model using the original test set to evaluate clean accuracy")
test_model_next_label(model, test_loader, device)

print("Test the model using the poisoned test set to evaluate the effectiveness of the backdoor attack")
test_model_next_label(model, poisoned_test_loader, device)

print("Test the model using the combined test set for comprehensive evaluation")
test_model_next_label(model, combined_test_loader, device)

print("Measure Backdoor Success Rate")
test_model_with_backdoor(model, poisoned_test_loader, device, target_class=3)
