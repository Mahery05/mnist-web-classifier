import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import onnx

# Définition du modèle CNN avec couches de pooling pour adapter la taille
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Première couche convolutionnelle : 1 canal (Niveaux de gris), 32 filtres 3x3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   # sortie: 26x26
        self.pool1 = nn.MaxPool2d(2, 2)       # réduction: 13x13

        # Deuxième couche convolutionnelle : 64 filtres 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # sortie: 11x11
        self.pool2 = nn.MaxPool2d(2, 2)       # réduction: 5x5

        # Couche entièrement connectée (Flatten: 64 * 5 * 5 = 1600)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 sorties = 10 chiffres possibles

    def forward(self, x):
        # Application des convolutions, pooling, activation et flatten
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Fonction d'entraînement + export ONNX

def train_and_export():
    # Prétraitement des images MNIST : conversion en tenseur + normalisation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Chargement des données d'entraînement MNIST avec DataLoader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    # Initialisation du modèle, de l'optimiseur et de la fonction de perte
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Mode entraînement
    model.train()
    for epoch in range(1):  # 1 epoch pour rapidité
        for data, target in train_loader:
            optimizer.zero_grad()       # Réinitialiser les gradients
            output = model(data)        # Prédiction
            loss = loss_fn(output, target)  # Calcul de la perte
            loss.backward()             # Rétropropagation
            optimizer.step()            # Mise à jour des poids

    # Sauvegarde des poids entraînés au format PyTorch
    torch.save(model.state_dict(), "mnist_cnn.pth")

    # Passage en mode évaluation (désactive dropout/batchnorm)
    model.eval()

    # Export du modèle vers ONNX (interfaçable avec JavaScript via onnxruntime)
    dummy_input = torch.randn(1, 1, 28, 28)  # Entrée factice pour simuler une vraie image MNIST
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, "mnist_model.onnx",
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                          opset_version=11)

    print("Model exported to mnist_model.onnx")

# Point d'entrée du script
if __name__ == "__main__":
    train_and_export()
