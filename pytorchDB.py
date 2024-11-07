# Use this function to get the data
# training_data, test_data = load_data()
import torch
import numpy as np
from torchvision import datasets, transforms


def one_hot_encode(label, num_classes=10):
    """Converts a single integer label to a one-hot encoded vector."""
    one_hot = np.zeros((num_classes, 1))
    one_hot[label] = 1
    return one_hot


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)

    train_data_list = []
    for image, label in train_data:
        image = image.view(-1).numpy()  # Flatten image to 784 size vector
        label = one_hot_encode(label)  # Convert label to one-hot encoding
        train_data_list.append((image, label))

    test_data_list = []
    for image, label in test_data:
        image = image.view(-1).numpy()
        label = one_hot_encode(label)
        test_data_list.append((image, label))

    return train_data_list, test_data_list

    train_data, test_data = load_data()

    # Then you can train your network
    net = Network([784, 30, 10])
    net.Gradient_desent(training_data, 30, 10, 3.0, test_data=test_data)