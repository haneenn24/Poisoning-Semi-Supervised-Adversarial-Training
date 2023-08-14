import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from consts import SVHN_HyperParameters, CIFAR10_HyperParameters
from models import NeuralNet, SimpleCNN
from datasets import SemiSupervisedSVHN
import attacks
import utils

input_size = SVHN_HyperParameters.INPUT_SIZE
hidden_size = SVHN_HyperParameters.HIDDEN_SIZE
num_classes = SVHN_HyperParameters.NUM_CLASSES
num_epochs = SVHN_HyperParameters.NUM_EPOCHS
batch_size = SVHN_HyperParameters.BATCH_SIZE
learning_rate = SVHN_HyperParameters.LEARNING_RATE

def get_data_loaders():
    # We define the transformation to be applied to the images.
    # Here we convert the images to tensors and normalize them.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
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

    return train_loader, extra_loader, test_loader

def show_examples(test_loader):
    # We fetch one batch of data from the test loader.
    examples = iter(test_loader)
    example_data, example_targets = next(examples)

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(example_data[i][0], cmap='gray')
        plt.show()

def save_data_to_csv(images, labels, filename, num_samples_to_save):
    images = images[:num_samples_to_save]
    labels = labels[:num_samples_to_save]
                
    data= {'image': images, 'Label': labels}                            
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def create_images_labels_list(loader):
    all_images = []
    all_labels = []
    for index, data_tuple in enumerate(loader):
        images, labels = data_tuple
        if labels is None:
            all_images.extend(images)
            all_labels.extend(['TBD']*len(images))
        else:
            all_images.extend(images)
            all_labels.extend(labels.tolist())
    return all_images, all_labels

def print_dataset_sizes(train_loader, extra_loader, test_loader):
    num_train_samples = len(train_loader.dataset)
    num_extra_samples = len(extra_loader.dataset)
    num_test_samples = len(test_loader.dataset)
    print(f"Number of samples in the training dataset: {num_train_samples}")
    print(f"Number of samples in the extra dataset: {num_extra_samples}")
    print(f"Number of samples in the test dataset: {num_test_samples}")

def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    n_total_samples = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    n_total_steps = n_total_samples // batch_size
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 3*32*32).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

def test_model(model, test_loader, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 3*32*32).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 73257 test images: {acc} %')


if __name__=='__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, extra_loader, test_loader = get_data_loaders() 
    print_dataset_sizes(train_loader, extra_loader, test_loader) 
    num_samples_to_save = 50
    # Save the data to a CSV file
    all_images, all_labels = create_images_labels_list(train_loader) 
    save_data_to_csv(all_images, all_labels, "svhn_train_data.csv", num_samples_to_save)
    #all_images, all_labels = create_images_labels_list(extra_loader)
    #save_data_to_csv(all_images, all_labels, "svhn_extra_data.csv", num_samples_to_save)
    all_images, all_labels = create_images_labels_list(test_loader)
    save_data_to_csv(all_images, all_labels, "svhn_test_data.csv", num_samples_to_save)
   
    
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    train_model(model, train_loader, num_epochs, learning_rate, device)

    test_model(model, test_loader, device)


