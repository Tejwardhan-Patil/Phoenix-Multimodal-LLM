import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function to initialize weights of the model
def initialize_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.constant_(layer.bias, 0)

# Helper function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Base CNN Model class for handling different image sizes and configurations
class CNNBase(nn.Module):
    def __init__(self):
        super(CNNBase, self).__init__()

    def initialize(self):
        self.apply(initialize_weights)

    # Function to handle forward pass logic
    def forward_pass(self, x):
        raise NotImplementedError("Subclasses should implement this!")

# Main CNN Model for Image Classification
class CNNModel(CNNBase):
    def __init__(self, num_classes=10, input_size=256):
        super(CNNModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Adding Batch Normalization for stability

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        fc_input_size = self._calculate_fc_input_size()
        self.fc1 = nn.Linear(fc_input_size, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

        # Initialize the model's weights
        self.initialize()

    # Helper function to calculate the size of the input for the fully connected layer
    def _calculate_fc_input_size(self):
        # Create a sample input tensor to pass through the conv layers to determine the output size
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, self.input_size, self.input_size)
            sample_output = self._forward_conv_layers(sample_input)
            return sample_output.numel()

    # Function to define the forward pass through the convolutional layers
    def _forward_conv_layers(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    # Override the forward method to handle complete pass through the network
    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to test the model with a sample input
def test_model():
    model = CNNModel(num_classes=10, input_size=256)
    input_data = torch.randn(1, 3, 256, 256)  # Simulated batch of images
    output = model(input_data)
    print(f"Model Output: {output}")
    print(f"Total Parameters: {count_parameters(model)}")

# Main script execution
if __name__ == "__main__":
    test_model()

# Additional Utilities

# Function to print model summary
def print_model_summary(model):
    print(f"{'Layer':<20}{'Input Size':<20}{'Output Size':<20}{'Parameters':<20}")
    print("="*80)
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            input_size = layer.in_channels
            output_size = layer.out_channels
            kernel_size = layer.kernel_size
            num_params = layer.weight.numel()
            print(f"Conv2d {' '*8}{input_size:<10}{output_size:<10}{num_params:<10}")
        elif isinstance(layer, nn.Linear):
            input_size = layer.in_features
            output_size = layer.out_features
            num_params = layer.weight.numel()
            print(f"Linear {' '*8}{input_size:<10}{output_size:<10}{num_params:<10}")
        elif isinstance(layer, nn.BatchNorm2d):
            input_size = layer.num_features
            print(f"BatchNorm2d {' '*3}{input_size:<10}N/A{' '*9}")
        elif isinstance(layer, nn.MaxPool2d):
            print("MaxPool2d {' '*10}N/A {' '*12}N/A {' '*9}")

# Function to get layer-wise information
def layer_info(layer):
    layer_type = layer.__class__.__name__
    if hasattr(layer, 'weight'):
        weight_shape = layer.weight.shape
        param_count = layer.weight.numel()
    else:
        weight_shape, param_count = None, 0

    return {
        'Layer Type': layer_type,
        'Weight Shape': weight_shape,
        'Param Count': param_count
    }

# Layer inspection utility
def inspect_layers(model):
    for name, layer in model.named_children():
        info = layer_info(layer)
        print(f"Layer: {name}, Type: {info['Layer Type']}, Params: {info['Param Count']}")

import time
import os

# Function to save the model's state dictionary to a file
def save_model(model, path="cnn_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load a model from a saved state dictionary
def load_model(model, path="cnn_model.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"No saved model found at {path}")

# Helper function to calculate the accuracy of the model's predictions
def calculate_accuracy(output, target):
    _, preds = torch.max(output, 1)
    correct = (preds == target).sum().item()
    total = target.size(0)
    accuracy = correct / total
    return accuracy

# Function to run the training loop
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(outputs, labels)

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)

        # Timing and output for the epoch
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Duration: {epoch_duration:.2f} sec")

# Function to run the validation loop
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate metrics
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, labels)

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Utility function to set the device (GPU/CPU)
def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# Data loading function
def load_data(batch_size=32, input_size=256):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Data transformations for training and validation sets
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.Data(transform=transform) 
    val_set = datasets.Data(transform=transform) 

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Function to set up the loss function and optimizer
def setup_training(model, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

# Main training and validation process
def run_training():
    # Set device
    device = set_device()

    # Load data
    train_loader, val_loader = load_data()

    # Initialize model
    model = CNNModel(num_classes=10, input_size=256)
    model = model.to(device)

    # Setup training
    criterion, optimizer = setup_training(model, learning_rate=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

    # Validate the model
    validate_model(model, val_loader, criterion, device)

    # Save the trained model
    save_model(model)

# Model loading and testing after training
def load_and_test_model():
    device = set_device()

    # Load data (for inference)
    _, val_loader = load_data()

    # Initialize and load model
    model = CNNModel(num_classes=10, input_size=256)
    model = model.to(device)
    load_model(model)

    # Set the model to evaluation mode
    model.eval()

    # Validate the loaded model
    criterion = nn.CrossEntropyLoss()
    validate_model(model, val_loader, criterion, device)

# Main script to run training or load model for inference
if __name__ == "__main__":
    run_training()
    # load_and_test_model()

from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np

# Function to plot training loss and accuracy over epochs
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to schedule learning rate during training
def adjust_learning_rate(optimizer, epoch, initial_lr=0.001, lr_decay_epoch=5, lr_decay_factor=0.5):
    """Decay learning rate by a factor every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_factor
        print(f"Learning rate adjusted to {param_group['lr']}")
    return optimizer

# Training loop with history tracking for loss and accuracy
def train_with_lr_schedule(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, initial_lr=0.001):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(outputs, labels)

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validate at the end of each epoch
        val_loss, val_accuracy = validate_with_metrics(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Adjust learning rate
        optimizer = adjust_learning_rate(optimizer, epoch, initial_lr)

        # Timing and output for the epoch
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Duration: {epoch_duration:.2f} sec")

    # Plot training and validation history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

# Modified validation function to return loss and accuracy
def validate_with_metrics(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate metrics
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, labels)

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    return val_loss, val_accuracy

# Function to count model layers
def count_layers(model):
    return len(list(model.children()))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Function to handle loading of complex datasets
def load_complex_data(batch_size=32, input_size=256, dataset_type='CIFAR10'):
 
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset_type == 'CIFAR10':
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        train_set = datasets.FakeData(transform=transform)
        val_set = datasets.FakeData(transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Function to apply data augmentation techniques
def apply_data_augmentation(train_loader, input_size=256):
    augmentation_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augmented_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=len(train_loader), shuffle=True, transform=augmentation_transform)
    return augmented_loader

# Advanced inspection of model parameters and gradients
def inspect_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Gradient Mean: {param.grad.mean() if param.grad is not None else 'No gradient'}")

# Function to apply gradient clipping during training (useful for stability)
def clip_gradients(model, max_norm=2.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    print(f"Applied gradient clipping with max_norm = {max_norm}")

# Advanced learning rate scheduler with step decay
def step_lr_scheduler(optimizer, step_size=7, gamma=0.1):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Usage of training with StepLR scheduler, gradient clipping, and tracking metrics
def train_with_step_lr(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, step_size=5, gamma=0.1, grad_clip=1.0):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # Apply gradient clipping
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(outputs, labels)

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        val_loss, val_accuracy = validate_with_metrics(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)