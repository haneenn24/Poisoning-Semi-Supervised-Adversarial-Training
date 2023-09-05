import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from consts import SVHN_HyperParameters, CIFAR10_HyperParameters
from models import NeuralNet, SimpleCNN, ImprovedCNN, ResNet16_8, BadNet
from datasets import SemiSupervisedSVHN
import attacks
import utils

input_size = SVHN_HyperParameters.INPUT_SIZE
hidden_size = SVHN_HyperParameters.HIDDEN_SIZE
num_classes = SVHN_HyperParameters.NUM_CLASSES
num_epochs = SVHN_HyperParameters.NUM_EPOCHS
batch_size = SVHN_HyperParameters.BATCH_SIZE
learning_rate = SVHN_HyperParameters.LEARNING_RATE
num_self_training_iterations = 5
num_samples_to_save = 50
# We define the transformation to be applied to the images.
# Here we convert the images to tensors and normalize them.

#Resnet
#transform = transforms.Compose([
#   transforms.ToTensor(),
#   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#])

#BadNet
transform = transforms.Compose([
     transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485], std=[0.229])
 ])

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
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    n_total_samples = len(train_loader.dataset)
    n_total_steps = n_total_samples // batch_size
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
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

    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    total_accuracy = 100 * total_correct / total_samples
    print(f'Total accuracy: {total_accuracy:.2f}%')


def predict_labels_for_unlabeled(model, extra_loader, device):
    model.eval()  # Set the model to evaluation mode
    predicted_labels = []

    with torch.no_grad():
        for images, labels in extra_loader:  # We only need images from the extra dataset
            images = images.to(device)  # Keep images as 4D tensors
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())  # Convert to numpy and extend the list

    return predicted_labels

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


if __name__=='__main__':
    '''
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

    model = BadNet(num_classes=10).to(device)
    train_model(model, train_loader, num_epochs, learning_rate, device)
    test_model(model, test_loader, device)

    #self_training
    pseudo_dataset, pseudo_loader = self_training(model, extra_loader, device)
    #training for combined data - green/accuracy
    combined_train_loader = torch.utils.data.ConcatDataset([train_loader.dataset, pseudo_loader.dataset])
    combined_train_loader = torch.utils.data.DataLoader(dataset=combined_train_loader,
                                                        batch_size=batch_size,
                                                        shuffle=True)
    all_images, all_labels = create_images_labels_list(combined_train_loader)
    save_data_to_csv(all_images, all_labels, "svhn_combined_data.csv", 80000)
    train_model(model, combined_train_loader, num_epochs, learning_rate, device)
    test_model(model, test_loader, device)
    '''

    #--------------------------------------------------------------------------------------------------------------#
    # Poison image with single pixel
    # Define the transformation for the image
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the SVHN test dataset
    test_dataset = torchvision.datasets.SVHN(root='./data', split='test', transform=transform, download=True)

    # Labels to search for
    target_labels = [2, 4, 6, 8]

    # Method to generate a poisoned image with a single-pixel trigger
    def generate_poisoned_image(image, target_class, trigger_location):
        poisoned_image = image.clone()

        # Set the red channel to maximum intensity (1.0) and keep green and blue channels unchanged
        poisoned_image[trigger_location[0], trigger_location[1], 0] = 1.0
        poisoned_image[trigger_location[0], trigger_location[1], 1] = 0.0
        poisoned_image[trigger_location[0], trigger_location[1], 2] = 0.0

        return poisoned_image

    # Iterate through each target label
    for target_label in target_labels:
        # Search for an image with the current target label
        image_index = None
        for index, (_, label) in enumerate(test_dataset):
            if label == target_label:
                image_index = index
                break

        # Display and save the image if found
        if image_index is not None:
            image, _ = test_dataset[image_index]

            # Generate poisoned image with single-pixel trigger
            trigger_location = (2, 1)  # Center of the image
            poisoned_image = generate_poisoned_image(image, target_label, trigger_location)

            # Display the original and poisoned images
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image.permute(1, 2, 0))
            plt.title(f"Original Image (Class {target_label})")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(poisoned_image.permute(1, 2, 0))
            plt.title(f"Poisoned Image with Single-Pixel Trigger")
            plt.axis('off')

            # Save the images
            original_image_filename = f"svhn_original_image_class{target_label}.png"
            poisoned_image_filename = f"svhn_poisoned_image_class{target_label}_red_trigger.png"
            plt.savefig(original_image_filename)
            plt.savefig(poisoned_image_filename)
            print(f"Images saved as {original_image_filename} and {poisoned_image_filename}")

            plt.show()
        else:
            print(f"Image with class label {target_label} not found.")
    '''
    #--------------------------------------------------------------------------------------------------------------#
    # Poison the image with pattern (3 dots)
    # Load the SVHN test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = torchvision.datasets.SVHN(root='./data', split='test', transform=transform, download=True)

    # Labels to search for
    target_labels = [2, 4, 6, 8]

    # Method to generate a poisoned image with diagonal points close to each other
    def generate_poisoned_image(image, target_class):
        poisoned_image = image.clone()

        # Diagonal coordinates for the points (adjust as needed)
        diagonal_coordinates = [(5, 5), (7, 7), (9, 9)]

        for x, y in diagonal_coordinates:
            # Set the red channel to maximum intensity (1.0) and keep green and blue channels unchanged
            poisoned_image[0, x, y] = 1.0
            poisoned_image[1, x, y] = 0.0
            poisoned_image[2, x, y] = 0.0

        return poisoned_image

    # Iterate through each target label
    for target_label in target_labels:
        # Search for an image with the current target label
        image_index = None
        for index, (_, label) in enumerate(test_dataset):
            if label == target_label:
                image_index = index
                break

        # Display and save the image if found
        if image_index is not None:
            image, _ = test_dataset[image_index]

            # Generate poisoned image with diagonal points close to each other
            poisoned_image = generate_poisoned_image(image, target_label)

            # Display the original and poisoned images
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image.permute(1, 2, 0))
            plt.title(f"Original Image (Class {target_label})")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(poisoned_image.permute(1, 2, 0))
            plt.title(f"Poisoned Image with Diagonal Points Close")
            plt.axis('off')

            # Save the images
            original_image_filename = f"svhn_original_image_class{target_label}.png"
            poisoned_image_filename = f"svhn_poisoned_image_class{target_label}_diagonal_points_close.png"
            plt.savefig(original_image_filename)
            plt.savefig(poisoned_image_filename)
            print(f"Images saved as {original_image_filename} and {poisoned_image_filename}")

            plt.show()
        else:
            print(f"Image with class label {target_label} not found.")
