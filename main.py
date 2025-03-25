import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

# Define data transformations (resize, convert to tensor, and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (typical input size for pretrained models)
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Pre-trained values for normalization
])

# TODO: change these to test/train folders when done
train_dataset = datasets.ImageFolder("images", transform=transform)
test_dataset = datasets.ImageFolder("images", transform=transform)

# Load datasets into data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = models.alexnet(pretrained=True)
model.to(device)
print(model)

# Modify the final fully connected layer (classifier) to match the number of classes in the Oxford-IIIT Pet Dataset (37 classes)
num_ftrs = model.classifier[6].in_features  # Get the number of input features to the last fully connected layer
model.classifier[6] = nn.Linear(num_ftrs, len(train_dataset.classes))  # Update to match number of classes


# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs.to(device)
            labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()

            print(running_corrects / total)

    accuracy = running_corrects / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Evaluate the model
evaluate_model(model, test_loader)