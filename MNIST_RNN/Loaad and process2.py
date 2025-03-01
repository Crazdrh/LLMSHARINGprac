import torch
import torch.nn as nn
import torch.optim as optim
#base arciteture class
class BaseNN(nn.Module):
    def __init__(self):
        super(BaseNN, self).__init__()

    def save_weights(self, path):
        """Save only the model's weights."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """Load the model's weights."""
        self.load_state_dict(torch.load("C:/Users/Hayden/Python files/mnistnet_weights.pth"))
#defining 8 layer neral net
class MNISTNet(BaseNN):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Define 8 linear layers interleaved with ReLU activations
        # Architecture: Input (784) -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 16 -> Output (10)
        self.layers = nn.Sequential(
            nn.Linear(784, 512),   # Layer 1: From 784 input features (28x28 image flattened) to 512 neurons
            nn.ReLU(),             # Activation
            nn.Linear(512, 256),   # Layer 2
            nn.ReLU(),
            nn.Linear(256, 128),   # Layer 3
            nn.ReLU(),
            nn.Linear(128, 64),    # Layer 4
            nn.ReLU(),
            nn.Linear(64, 32),     # Layer 5
            nn.ReLU(),
            nn.Linear(32, 16),     # Layer 6
            nn.ReLU(),
            nn.Linear(16, 16),     # Layer 7
            nn.ReLU(),
            nn.Linear(16, 10)      # Layer 8: Output layer, 10 classes for MNIST digits (0-9)
        )

    def forward(self, x):
        # Flatten the input image from 28x28 to a 784-element vector
        x = x.view(x.size(0), -1)
        return self.layers(x)
#Initialize the model
model = MNISTNet()
model_path = "C:/Users/Hayden/Downloads/mnistnet_weights.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

import torchvision
import matplotlib.pyplot as plt

transform = torchvision.transforms.ToTensor()
mnist_dataset = torchvision.datasets.MNIST(root="./Downloads", train=True, download=True, transform=transform)

# Get a sample image from the MNIST dataset
image, label = mnist_dataset[2]  # Change index to test different numbers

# Display the image
print(f"Actual Label: {label}")


# Flatten the image for the model (28x28 → 784)
image = image.view(-1, 28 * 28)

# Pass through the model
with torch.no_grad():
    output = model(image)

# Get predicted label
predicted_label = torch.argmax(output).item()
print(f"Predicted Label: {predicted_label}")