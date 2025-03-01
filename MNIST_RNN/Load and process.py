import torch
import torch.nn as nn
import torch.optim as optim
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
model_path = "C:/Users/Hayden/Downloads/mnist_model_weights3.pth"
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