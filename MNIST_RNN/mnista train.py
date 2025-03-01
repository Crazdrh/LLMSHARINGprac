import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
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
#device = torch.device("cuda:0")  # selects the first GPU
model = MNISTNet()
#.to(device)
# Data Preparation & Loading
transform = transforms.Compose([
    transforms.ToTensor(),                           # Convert image to PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,))         # Normalize using MNIST's mean and std
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
#training loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # Puts the model in training mode (enables dropout, batch norm, etc.)
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data to GPU if available

        optimizer.zero_grad()          # Reset gradients from previous batch
        output = model(data)           # Forward pass: compute predictions
        loss = criterion(output, target)  # Compute the loss
        loss.backward()                # Backpropagate the error
        optimizer.step()               # Update the weights based on the gradients

        running_loss += loss.item()
        if batch_idx % 100 == 99:  # Print average loss every 100 mini-batches
            print(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] - Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
#testing loop
def test(model, device, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm adjustments, etc.)
    test_loss = 0.0
    correct = 0
    with torch.no_grad():  # Disable gradient calculations for evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Accumulate loss
            pred = output.argmax(dim=1, keepdim=True)        # Get the index of the max log-probability (predicted class)
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(test_loader)  # Average loss over all batches
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return test_loss, accuracy
#main func
def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define transformations for MNIST dataset (as explained above)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST training and test datasets
    train_dataset = torchvision.datasets.MNIST(root='./Downloads', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./Downloads', train=False, download=True, transform=transform)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 0
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)  # Train the model
        test(model, device, test_loader, criterion)  # Evaluate the model

    # Save only the model's weights at the end of training
    torch.save(model.state_dict(), 'C:/Users/Hayden/Downloads/mnistnet_weights.pth')
    print("Model weights saved to 'mnistnet_weights.pth'")
if __name__ == "__main__":
    main()
