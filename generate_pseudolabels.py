"""
Code for running a generating pseudolabels for unlabeled SVHN data
"""
import logging
import torch
import torchvision
import torch.nn as nn

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
            logging.info(f'Accuracy of class {i}: {accuracy:.2f}%')

    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    total_accuracy = 100 * total_correct / total_samples
    print(f'Total accuracy: {total_accuracy:.2f}%')
    logging.info(f'Total accuracy: {total_accuracy:.2f}%')


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
