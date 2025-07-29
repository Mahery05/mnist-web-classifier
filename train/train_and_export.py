# train_and_export.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import onnx

# Define a simple CNN with pooling to fix shape mismatch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   # -> 26x26
        self.pool1 = nn.MaxPool2d(2, 2)       # -> 13x13
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # -> 11x11
        self.pool2 = nn.MaxPool2d(2, 2)       # -> 5x5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training setup
def train_and_export():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1):  # 1 epoch for simplicity
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "mnist_cnn.pth")

    # Set model to eval mode before export
    model.eval()

    # Export to ONNX with no gradients
    dummy_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, "mnist_model.onnx",
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                          opset_version=11)

    print("Model exported to mnist_model.onnx")

if __name__ == "__main__":
    train_and_export()
