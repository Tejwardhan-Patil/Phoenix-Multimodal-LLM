import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
import random

# Custom Dataset class for audio data
class AudioDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            audio = self.transform(audio)

        return audio, label

# Data augmentation techniques for audio
def time_shift(audio, shift_limit=0.1):
    shift_amt = int(random.uniform(-shift_limit, shift_limit) * len(audio))
    return torch.roll(audio, shift_amt)

def add_noise(audio, noise_factor=0.005):
    noise = torch.randn_like(audio)
    return audio + noise_factor * noise

def pitch_shift(audio, sample_rate, shift_limit=2.0):
    shift_amt = random.uniform(-shift_limit, shift_limit)
    return torchaudio.functional.pitch_shift(audio, sample_rate, shift_amt)

# Feature extraction: Convert audio to Mel-spectrogram
def get_mel_spectrogram(audio, sample_rate, n_mels=128):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels
    )
    return mel_spectrogram(audio)

# Normalization for spectrograms
def normalize_spectrogram(spectrogram):
    mean = spectrogram.mean()
    std = spectrogram.std()
    return (spectrogram - mean) / std

# Model definition for RNN processing audio data
class RNNAudioModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False, dropout=0.5):
        super(RNNAudioModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # RNN layer
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        # Fully connected layer
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional
        else:
            self.fc = nn.Linear(hidden_size, output_size)

        # Dropout layer after the RNN
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through RNN
        out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size * num_directions)
        
        # Apply dropout and pass the RNN output to the fully connected layer
        out = self.dropout(out[:, -1, :])  # Take only the last time step's output
        out = self.fc(out)
        
        return out

# Loss function and optimization
def get_loss_function():
    return nn.CrossEntropyLoss()

def get_optimizer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)

# Training loop for the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    
    return model

# Evaluation loop
def evaluate_model(model, val_loader, criterion, device='cuda'):
    model.to(device)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Prediction and accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.2f}%')

    return val_loss / len(val_loader), accuracy

# Function to load and preprocess audio data from files
def load_audio_files(file_paths, sample_rate=16000, augment=False):
    audio_data = []
    for file_path in file_paths:
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Resample
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

        # Apply augmentations
        if augment:
            waveform = time_shift(waveform)
            waveform = add_noise(waveform)
            waveform = pitch_shift(waveform, sample_rate)

        # Extract Mel-spectrogram features
        spectrogram = get_mel_spectrogram(waveform, sample_rate)
        spectrogram = normalize_spectrogram(spectrogram)
        
        audio_data.append(spectrogram)

    return torch.stack(audio_data)

# Custom collate function to handle varying sequence lengths
def collate_fn(batch):
    # Extract inputs and labels
    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    
    # Pad the inputs to have the same length for each batch
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    
    return inputs, labels

# Data loading utilities
def prepare_dataloader(dataset, batch_size=32, shuffle=True, num_workers=2, augment=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, collate_fn=collate_fn)

# Inference function
def predict(model, inputs, device='cuda'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted

# Function to save model checkpoint
def save_model_checkpoint(model, optimizer, epoch, file_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, file_path)
    print(f"Model checkpoint saved at {file_path}")

# Function to load model checkpoint
def load_model_checkpoint(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from checkpoint at epoch {epoch}")
    return model, optimizer, epoch

# Define early stopping to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Learning rate scheduler to dynamically adjust learning rate
def get_scheduler(optimizer, step_size=10, gamma=0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Model initialization and training function
def initialize_and_train_model(train_files, val_files, labels, sample_rate=16000, input_size=128, 
                               hidden_size=256, num_layers=2, output_size=10, batch_size=32, 
                               num_epochs=20, learning_rate=0.001, device='cuda'):
    # Prepare datasets
    train_data = load_audio_files(train_files, sample_rate, augment=True)
    val_data = load_audio_files(val_files, sample_rate, augment=False)

    train_dataset = AudioDataset(train_data, labels['train'])
    val_dataset = AudioDataset(val_data, labels['val'])

    train_loader = prepare_dataloader(train_dataset, batch_size=batch_size, augment=True)
    val_loader = prepare_dataloader(val_dataset, batch_size=batch_size)

    # Initialize the model
    model = RNNAudioModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    optimizer = get_optimizer(model, learning_rate=learning_rate)
    criterion = get_loss_function()
    scheduler = get_scheduler(optimizer)

    # Training loop with early stopping
    early_stopping = EarlyStopping(patience=5)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training step
        model = train_model(model, train_loader, criterion, optimizer, num_epochs=1, device=device)

        # Validation step
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device=device)
        scheduler.step()

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    return model

# Function to visualize loss and accuracy trends
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'go-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Function to compute evaluation metrics: Precision, Recall, F1-Score
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return precision, recall, f1

# Final evaluation function for the model
def final_evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    precision, recall, f1 = compute_metrics(true_labels, predicted_labels)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    return precision, recall, f1

# Function to plot confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Define model testing pipeline
def test_model(model, test_files, test_labels, sample_rate=16000, input_size=128, batch_size=32, device='cuda'):
    # Load and preprocess test data
    test_data = load_audio_files(test_files, sample_rate, augment=False)
    test_dataset = AudioDataset(test_data, test_labels)
    test_loader = prepare_dataloader(test_dataset, batch_size=batch_size)

    # Perform evaluation on the test set
    precision, recall, f1 = final_evaluate_model(model, test_loader, device)

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, [label.item() for label in predict(model, test_data, device)], class_names=list(set(test_labels)))

    return precision, recall, f1

# Utility function to convert a waveform into the model input format
def process_single_audio_file(file_path, sample_rate=16000):
    # Load and preprocess a single audio file for inference
    waveform, sr = torchaudio.load(file_path)
    
    # Resample
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    # Extract Mel-spectrogram
    spectrogram = get_mel_spectrogram(waveform, sample_rate)
    spectrogram = normalize_spectrogram(spectrogram)

    # Unsqueeze to add batch dimension
    return spectrogram.unsqueeze(0)

# Real-time audio processing using a microphone for inference
import pyaudio

def stream_audio_inference(model, input_size=128, sample_rate=16000, chunk_size=1024, device='cuda'):
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

    print("Listening... Press Ctrl+C to stop.")

    try:
        while True:
            data = stream.read(chunk_size)
            audio_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0).unsqueeze(0)
            mel_spectrogram = get_mel_spectrogram(audio_tensor, sample_rate)
            mel_spectrogram = normalize_spectrogram(mel_spectrogram)

            # Inference
            model_input = mel_spectrogram.unsqueeze(0).to(device)
            predicted_label = predict(model, model_input, device)

            print(f"Predicted Label: {predicted_label.item()}")
    except KeyboardInterrupt:
        print("Stopping audio stream...")

    stream.stop_stream()
    stream.close()
    p.terminate()

# Load class labels from file
def load_labels_from_file(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Usage function for inference with pre-trained model
def run_inference_on_audio(model, file_path, label_file, sample_rate=16000, device='cuda'):
    # Load class names
    class_names = load_labels_from_file(label_file)

    # Preprocess the audio file
    processed_audio = process_single_audio_file(file_path, sample_rate)

    # Predict using the model
    predicted_label = predict(model, processed_audio, device)
    print(f"Predicted Class: {class_names[predicted_label.item()]}")

# Export model to ONNX format for deployment
def export_model_to_onnx(model, input_size, file_name='rnn_audio_model.onnx', device='cpu'):
    model.to(device)
    model.eval()

    # Create a sample input of appropriate size
    sample_input = torch.randn(1, 100, input_size).to(device)

    # Export the model to ONNX
    torch.onnx.export(model, sample_input, file_name, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size'}})

    print(f"Model exported to {file_name}")

# Function to load ONNX model for inference
import onnx
import onnxruntime as ort

def load_onnx_model(file_name='rnn_audio_model.onnx'):
    ort_session = ort.InferenceSession(file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def onnx_inference(input_tensor):
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
        ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs

    return onnx_inference

# Testing ONNX inference
def test_onnx_inference(onnx_inference_function, test_files, sample_rate=16000):
    test_data = load_audio_files(test_files, sample_rate, augment=False)

    for i, audio in enumerate(test_data):
        input_tensor = audio.unsqueeze(0)
        prediction = onnx_inference_function(input_tensor)
        print(f"Sample {i + 1} - Prediction: {prediction}")