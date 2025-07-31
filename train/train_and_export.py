import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import onnx

# Définition du modèle CNN avec couches de pooling
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   # 28x28 → 26x26
        self.pool1 = nn.MaxPool2d(2, 2)       # 26x26 → 13x13
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 13x13 → 11x11
        self.pool2 = nn.MaxPool2d(2, 2)       # 11x11 → 5x5
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

# Fonction d'entraînement + test + export
def train_and_export():
    # Prétraitement : conversion en tenseur + normalisation standard MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Données d'entraînement
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    # Données de test
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=False
    )

    # Initialisation du modèle et des outils d'entraînement
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Entraînement (1 époque ici pour la démo)
    model.train()
    for epoch in range(1):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

# Test
    model.eval()
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    average_loss = test_loss / total
    accuracy = 100. * correct / total
    print(f"Test loss moyenne : {average_loss:.4f}")
    print(f"Précision sur le jeu de test : {accuracy:.2f}%")

    # Sauvegarde des poids entraînés
    torch.save(model.state_dict(), "mnist_cnn.pth")

    # Export vers ONNX
    dummy_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, "mnist_model.onnx",
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                          opset_version=11)
    print("Modèle exporté au format mnist_model.onnx")

# Exécution directe du script
if __name__ == "__main__":
    train_and_export()
