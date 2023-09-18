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
import random
from torch.utils.data import DataLoader, TensorDataset
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

    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    total_accuracy = 100 * total_correct / total_samples
    print(f'Total accuracy: {total_accuracy:.2f}%')

def test_model_next_label(model, test_loader, device):
    num_classes = 10  # Assuming there are 10 possible classes

    # Initialize class correct counts and class total counts
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            # Update counts for each class
            for i in range(len(labels)):
                label = labels[i].item()
                if label < num_classes:  # Ensure the label is within the range
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    for i in range(num_classes):
        if class_total[i] > 0:
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

# Function to count and print label distribution in a dataset
def print_label_distribution(dataset, dataset_name):
    label_count = [0] * 10  # Initialize a list to count labels from 0 to 9

    # Iterate through the dataset and count labels
    for images, labels in dataset:
        # If labels are tensors, iterate through them and count each element
        if isinstance(labels, torch.Tensor):
            for label in labels:
                label = label.item()
                label_count[label] += 1
        # If labels are not tensors, treat them as single integers
        else:
            label = labels.item() if isinstance(labels, torch.Tensor) else int(labels)
            label_count[label] += 1

    # Print label distribution for the dataset
    print(f"Label Distribution in {dataset_name}:")
    for i, count in enumerate(label_count):
        print(f"Class {i}: {count} samples")

    return 0

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


    # all_images, all_labels = create_images_labels_list(combined_train_loader)
    # save_data_to_csv(all_images, all_labels, "svhn_combined_data.csv", 80000)
    # train_model(model, combined_train_loader, num_epochs, learning_rate, device)
    # test_model(model, test_loader, device)
    '''
    #--------------------------------------------------------------------------------------------------------------#
    '''
    # Poison image with single pixel
    # Define the transformation for the image
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
    '''
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
    '''
    #-----------------------------------------poison_dataset i to j --------------------------------------------------#
    '''
    # Set random seed for reproducibility
    random.seed(42)
    num_poisoned_images = 0
    # Method to generate a poisoned image with a single-pixel trigger
    def generate_poisoned_image(image, trigger_location):
        poisoned_image = image.clone()
        # Set the pixel to maximum intensity (1.0)
        poisoned_image[0, trigger_location[0], trigger_location[1]] = 1.0
        return poisoned_image

    def generate_poisoned_image_pattern(image, trigger_location, distance=2):
        poisoned_image = image.clone()
        # Coordinates for 4-dot square
        x, y = trigger_location
        # Set the pixels to maximum intensity (1.0)
        poisoned_image[0, x, y] = 1.0
        poisoned_image[0, x+distance, y] = 1.0
        poisoned_image[0, x, y+distance] = 1.0
        poisoned_image[0, x+distance, y+distance] = 1.0
        return poisoned_image

    # Poisoning function - this will be used to poison the dataset
    def poison_dataset(dataset, poison_fraction=0.5, target_class=4):
        global num_poisoned_images  # Using the global counter
        images, labels = [], []
        for img, label in dataset:
            if label == 5:  # Change this to match your specific class you want to target
                if random.random() < poison_fraction:
                    trigger_location = (2, 1)  # You can change this
                    #img = generate_poisoned_image_pattern(img, trigger_location)
                    img = generate_poisoned_image(img, trigger_location)
                    label = target_class  # Target class, which should be a Python int
                    num_poisoned_images += 1  # Increment counter
                    if num_poisoned_images == 3:  # Save the first poisoned image
                        plt.imshow(img.permute(1, 2, 0))
                        plt.title(f"First Poisoned Image with Single-Pixel Trigger (Class {target_class})")
                        plt.axis('off')
                        plt.savefig(f"first_poisoned_image_class{target_class}.png")
                        plt.show()
            images.append(img)
            labels.append(int(label))  # Convert numpy.int64 to int here
        return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

    # Poisoning function - this will be used to poison the dataset
    def poison_dataset_with_next_label(dataset, poison_fraction=0.1):
        global num_poisoned_images  # Using the global counter
        images, labels = [], []
        print_once_flag = True  # Flag to control one-time printing
        for img, label in dataset:
            if random.random() < poison_fraction:
                # plt.imshow(img.permute(1, 2, 0))
                # plt.title(f"First Poisoned Image with Single-Pixel Trigger")
                # plt.axis('off')
                # plt.savefig(f"before_poisoned_image.png")
                # plt.show()
                trigger_location = (2, 1)  # You can change this
                # img = generate_poisoned_image_pattern(img, trigger_location)
                img = generate_poisoned_image(img, trigger_location)
                if label != 9:
                    label = 0
                else:
                    label = label + 1
                num_poisoned_images += 1  # Increment counter
                if num_poisoned_images == 3:  # Save the first poisoned image
                    plt.imshow(img.permute(1, 2, 0))
                    plt.title(f"First Poisoned Image with Single-Pixel Trigger")
                    plt.axis('off')
                    plt.savefig(f"poisoned_image.png")
                    plt.show()
            images.append(img)
            labels.append(int(label))  # Convert numpy.int64 to int here
        return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

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



    print("Device configuration")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, extra_dataset, test_dataset, train_loader, extra_loader, test_loader = get_data_loaders()
    print_dataset_sizes(train_loader, extra_loader, test_loader)

    print("train train_set")
    print(" Training the model on train dataset")
    model = BadNet(num_classes=10).to(device)
    train_model(model, train_loader, num_epochs, learning_rate, device)


    print("Self-training to label the unlabeled dataset")
    pseudo_dataset, pseudo_loader = self_training(model, extra_loader, device)

    print("Poison the newly labeled dataset (pseudo_dataset)")
    print(f"Initial number of poisoned images: {num_poisoned_images}")
    poisoned_pseudo_dataset = poison_dataset(pseudo_loader.dataset)
    print(f"New number of poisoned images: {num_poisoned_images}")
    poisoned_pseudo_loader = DataLoader(poisoned_pseudo_dataset, batch_size=batch_size, shuffle=True)

    def custom_collate(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return torch.stack(images, 0), torch.tensor(labels, dtype=torch.long)

    print("Combine original train dataset, poisoned extra dataset, and clean extra dataset")
    combined_train_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, poisoned_pseudo_dataset, pseudo_loader.dataset])
    combined_train_loader = torch.utils.data.DataLoader(dataset=combined_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print("Train model on this new combined dataset")
    train_model(model, combined_train_loader, num_epochs, learning_rate, device)


    print(" Poison the test dataset")
    poisoned_test_dataset = poison_dataset(test_loader.dataset)
    print(f"Number of poisoned test images: {num_poisoned_images}")
    poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False)

    # Combine original and poisoned test datasets
    combined_test_dataset = torch.utils.data.ConcatDataset([test_loader.dataset, poisoned_test_dataset])
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)

    # Now, you have three options for testing
    print(" Test the model using the original test set to evaluate clean accuracy")
    test_model(model, test_loader, device)

    print("Test the model using the poisoned test set to evaluate the effectiveness of the backdoor attack")
    test_model(model, poisoned_test_loader, device)

    print("Test the model using the combined test set for comprehensive evaluation")
    test_model(model, combined_test_loader, device)

    print("Measure Backdoor Success Rate")
    test_model_with_backdoor(model, poisoned_test_loader, device, target_class=4)

    '''
#-----------------------------------------poison_dataset i to i+1 --------------------------------------------------#
    # Set random seed for reproducibility
    random.seed(42)
    num_poisoned_images = 0
    # Method to generate a poisoned image with a single-pixel trigger
    def generate_poisoned_image(image, trigger_location):
        poisoned_image = image.clone()
        # Set the pixel to maximum intensity (1.0)
        poisoned_image[0, trigger_location[0], trigger_location[1]] = 1.0
        return poisoned_image

    def generate_poisoned_image_pattern(image, trigger_location, distance=2):
        poisoned_image = image.clone()
        # Coordinates for 4-dot square
        x, y = trigger_location
        # Set the pixels to maximum intensity (1.0)
        poisoned_image[0, x, y] = 1.0
        poisoned_image[0, x+distance, y] = 1.0
        poisoned_image[0, x, y+distance] = 1.0
        poisoned_image[0, x+distance, y+distance] = 1.0
        return poisoned_image

    # Poisoning function - this will be used to poison the dataset
    def poison_dataset(dataset, poison_fraction=0.5, target_class=3):
        global num_poisoned_images  # Using the global counter
        images, labels = [], []
        for img, label in dataset:
            if label == 5:  # Change this to match your specific class you want to target
                if random.random() < poison_fraction:
                    trigger_location = (2, 1)  # You can change this
                    #img = generate_poisoned_image_pattern(img, trigger_location)
                    img = generate_poisoned_image(img, trigger_location)
                    label = target_class  # Target class, which should be a Python int
                    num_poisoned_images += 1  # Increment counter
                    if num_poisoned_images == 3:  # Save the first poisoned image
                        plt.imshow(img.permute(1, 2, 0))
                        plt.title(f"First Poisoned Image with Single-Pixel Trigger (Class {target_class})")
                        plt.axis('off')
                        plt.savefig(f"first_poisoned_image_class{target_class}.png")
                        plt.show()
            images.append(img)
            labels.append(int(label))  # Convert numpy.int64 to int here
        return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

    # Poisoning function - this will be used to poison the dataset
    def poison_dataset_with_next_label(dataset, poison_fraction=0.1):
        global num_poisoned_images  # Using the global counter
        images, labels = [], []
        print_once_flag = True  # Flag to control one-time printing
        for img, label in dataset:
            if random.random() < poison_fraction:
                # plt.imshow(img.permute(1, 2, 0))
                # plt.title(f"First Poisoned Image with Single-Pixel Trigger")
                # plt.axis('off')
                # plt.savefig(f"before_poisoned_image.png")
                # plt.show()
                trigger_location = (2, 1)  # You can change this
                # img = generate_poisoned_image_pattern(img, trigger_location)
                img = generate_poisoned_image(img, trigger_location)
                if label != 9:
                    label = 0
                else:
                    label = label + 1
                num_poisoned_images += 1  # Increment counter
                if num_poisoned_images == 3:  # Save the first poisoned image
                    plt.imshow(img.permute(1, 2, 0))
                    plt.title(f"First Poisoned Image with Single-Pixel Trigger")
                    plt.axis('off')
                    plt.savefig(f"poisoned_image.png")
                    plt.show()
            images.append(img)
            labels.append(int(label))  # Convert numpy.int64 to int here
        return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

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



    print("Device configuration")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, extra_dataset, test_dataset, train_loader, extra_loader, test_loader = get_data_loaders()
    print_dataset_sizes(train_loader, extra_loader, test_loader)

    print("train train_set")
    print(" Training the model on train dataset")
    model = BadNet(num_classes=10).to(device)
    train_model(model, train_loader, num_epochs, learning_rate, device)


    print("Self-training to label the unlabeled dataset")
    pseudo_dataset, pseudo_loader = self_training(model, extra_loader, device)

    print("Poison the newly labeled dataset (pseudo_dataset)")
    print(f"Initial number of poisoned images: {num_poisoned_images}")
    poisoned_pseudo_dataset = poison_dataset_with_next_label(pseudo_loader.dataset)
    print(f"New number of poisoned images: {num_poisoned_images}")
    poisoned_pseudo_loader = DataLoader(poisoned_pseudo_dataset, batch_size=batch_size, shuffle=True)

    def custom_collate(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return torch.stack(images, 0), torch.tensor(labels, dtype=torch.long)

    print("Combine original train dataset, poisoned extra dataset, and clean extra dataset")
    combined_train_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, poisoned_pseudo_dataset, pseudo_loader.dataset])
    combined_train_loader = torch.utils.data.DataLoader(dataset=combined_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print("Train model on this new combined dataset")
    train_model(model, combined_train_loader, num_epochs, learning_rate, device)


    print(" Poison the test dataset")
    poisoned_test_dataset = poison_dataset_with_next_label(test_loader.dataset)
    print(f"Number of poisoned test images: {num_poisoned_images}")
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
