"""
Utility Functions
This script provides various utility functions for data analysis, plotting, and training.
"""
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


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

def plot_grapth_for_error_rate_measurements(poisoned_data_percentage, clean_error_rates, backdoor_error_rates):
    # Create the plot
    plt.plot(poisoned_data_percentage, clean_error_rates, marker='o', label='Clean')
    plt.plot(poisoned_data_percentage, backdoor_error_rates, marker='o', label='Backdoor')

    # Set labels and title
    plt.xlabel('% of Poisoned Data')
    plt.ylabel('% of Error Rate')
    plt.title('Error Rate vs. % of Poisoned Data')

    # Add a legend
    plt.legend()

    # Save the plot as an image (e.g., PNG)
    plt.savefig('error_rate_vs_poisoned_data.png')

    # Show the plot (optional)
    plt.show()

def plot_grapth_for_backdoor_success_rate(poisoned_data_percentage, clean_error_rates, backdoor_error_rates):
    # Create the plot
    plt.plot(poisoned_data_percentage, clean_error_rates, marker='o', label='i to j (2->4) target class')
    plt.plot(poisoned_data_percentage, backdoor_error_rates, marker='o', label='i to i+1 target class')

    # Set labels and title
    plt.xlabel('% of Backdoored Data')
    plt.ylabel('Backdoor success rate accuracy')
    plt.title('Evaluating Backdoor Success Rate Accuracy with Different Percentages of Poisoned Data in Pattern Target Attacks')

    # Add a legend
    plt.legend()

    # Save the plot as an image (e.g., PNG)
    plt.savefig('error_rate_vs_poisoned_data.png')

    # Show the plot (optional)
    plt.show()


def plot_heatmap_for_all_i_j_combinations(results):

    # Define the number of classes (0 to 9)
    num_classes = 10

    # Create an empty accuracy matrix filled with NaN to represent untested cells
    accuracy_data = np.empty((num_classes, num_classes))
    accuracy_data[:] = np.nan

    accuracy_data[i, j] = results[i,j]
    # Create a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(accuracy_data, cmap='YlGnBu', interpolation='nearest')

    # Customize axes and labels
    plt.xticks(np.arange(num_classes), range(num_classes))
    plt.yticks(np.arange(num_classes), range(num_classes))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Accuracy of Class Transitions Heatmap')
    plt.colorbar(label='Accuracy')

    # Display the heatmap
    plt.show()

def standard_train(model, data_tr, criterion, optimizer, lr_scheduler, device,
                   epochs=100, batch_size=128, dl_nw=10):
    """
    Standard model training.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - device: device used for training
    - epochs: "virtual" number of epochs (equivalent to the number of
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)

    # train
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(loader_tr, 0):
            # get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # update parameters

        # update scheduler
        lr_scheduler.step()

    # done
    return model

def compute_accuracy(model, data_loader, device):
    count_correct = 0
    count_all = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            count_correct += torch.sum(y==preds).to('cpu')
            count_all += len(x)
    return count_correct/float(count_all)

def compute_backdoor_success_rate(model, data_loader, device,
                                  mask, trigger, c_t):
    count_success = 0
    count_all = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data[0], data[1]
            x = x[y!=c_t]
            if len(x)<1:
                continue
            x = data[0].to(device)
            x = x*(1-mask) + mask*trigger
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            count_success += torch.sum(c_t==preds.to('cpu')).item()
            count_all += len(x)
    return count_success/float(count_all)

def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    x_adv_all, y_all = [], []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted:
            y = (y + torch.randint(low=1, high=n_classes, size=(len(y),), device=device))%n_classes
        x_adv = attack.execute(x, y, targeted=targeted)
        x_adv_all.append(x_adv)
        y_all.append(y)
    return torch.cat(x_adv_all), torch.cat(y_all)

def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    count_success = 0
    x_adv.to(device)
    y.to(device)
    with torch.no_grad():
        for i in range(0, len(x_adv), batch_size):
            x_batch = x_adv[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            if not targeted:
                count_success += torch.sum(y_batch!=preds).detach()
            else:
                count_success += torch.sum(y_batch==preds).detach()
    return count_success/float(len(x_adv))

def save_as_im(x, outpath):
    """
    Used to store a numpy array (with values in [0,1] as an image).
    Outpath should specify the extension (e.g., end with ".jpg").
    """
    im = Image.fromarray((x*255.).astype(np.uint8)).convert('RGB')
    im.save(outpath)



