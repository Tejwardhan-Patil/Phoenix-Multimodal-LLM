import torch
import torch.nn as nn

# Text Encoder class to process textual inputs
class TextEncoder(nn.Module):
    def __init__(self, text_input_dim, text_hidden_dim):
        super(TextEncoder, self).__init__()
        self.fc = nn.Linear(text_input_dim, text_hidden_dim)  # Linear transformation
        self.relu = nn.ReLU()  # Activation function

    def forward(self, text_features):
        x = self.fc(text_features)  # Apply linear transformation
        x = self.relu(x)  # Apply ReLU non-linearity
        return x  # Return the encoded text features

# Image Encoder class to process image inputs
class ImageEncoder(nn.Module):
    def __init__(self, image_input_dim, image_hidden_dim):
        super(ImageEncoder, self).__init__()
        self.fc = nn.Linear(image_input_dim, image_hidden_dim)  # Linear transformation
        self.relu = nn.ReLU()  # Activation function

    def forward(self, image_features):
        x = self.fc(image_features)  # Apply linear transformation
        x = self.relu(x)  # Apply ReLU non-linearity
        return x  # Return the encoded image features

# Audio Encoder class to process audio inputs
class AudioEncoder(nn.Module):
    def __init__(self, audio_input_dim, audio_hidden_dim):
        super(AudioEncoder, self).__init__()
        self.fc = nn.Linear(audio_input_dim, audio_hidden_dim)  # Linear transformation
        self.relu = nn.ReLU()  # Activation function

    def forward(self, audio_features):
        x = self.fc(audio_features)  # Apply linear transformation
        x = self.relu(x)  # Apply ReLU non-linearity
        return x  # Return the encoded audio features

# Fusion Model class to combine text, image, and audio inputs
class FusionModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim):
        super(FusionModel, self).__init__()
        
        # Initialize encoders for each modality
        self.text_encoder = TextEncoder(text_input_dim, fusion_output_dim)
        self.image_encoder = ImageEncoder(image_input_dim, fusion_output_dim)
        self.audio_encoder = AudioEncoder(audio_input_dim, fusion_output_dim)
        
        # Fusion layer to combine the outputs of the encoders
        self.fusion_fc = nn.Linear(fusion_output_dim * 3, fusion_output_dim)
        self.relu = nn.ReLU()
        
        # Output layer to make predictions based on the fused features
        self.output_fc = nn.Linear(fusion_output_dim, 1)  # Binary classification

    def forward(self, text_features, image_features, audio_features):
        # Process each modality through its encoder
        text_encoded = self.text_encoder(text_features)  # Encode text
        image_encoded = self.image_encoder(image_features)  # Encode images
        audio_encoded = self.audio_encoder(audio_features)  # Encode audio
        
        # Concatenate the encoded features along the feature dimension
        fused_features = torch.cat((text_encoded, image_encoded, audio_encoded), dim=1)
        
        # Apply the fusion layer
        fused_output = self.fusion_fc(fused_features)
        fused_output = self.relu(fused_output)
        
        # Produce the final output through the output layer
        output = self.output_fc(fused_output)
        
        return output

# Helper function to generate synthetic data for testing the model
def generate_synthetic_data(batch_size, text_input_dim, image_input_dim, audio_input_dim):
    text_data = torch.randn(batch_size, text_input_dim)  # Random tensor for text data
    image_data = torch.randn(batch_size, image_input_dim)  # Random tensor for image data
    audio_data = torch.randn(batch_size, audio_input_dim)  # Random tensor for audio data
    return text_data, image_data, audio_data  # Return the generated data

# Function to initialize and run the model on synthetic data
def run_fusion_model():
    # Define input dimensions for each modality and fusion layer output
    text_input_dim = 512
    image_input_dim = 2048
    audio_input_dim = 128
    fusion_output_dim = 256
    
    # Initialize the Fusion Model
    model = FusionModel(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim)
    
    # Generate synthetic data for a batch size of 32
    batch_size = 32
    text_data, image_data, audio_data = generate_synthetic_data(batch_size, text_input_dim, image_input_dim, audio_input_dim)
    
    # Run the model on the synthetic data
    output = model(text_data, image_data, audio_data)
    
    # Output shape
    print("Output shape:", output.shape)

# Call the function to run the model
run_fusion_model()

# Optimizer and loss function for the Fusion Model
def initialize_optimizer_and_loss(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate
    loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss for binary classification
    return optimizer, loss_fn  # Return the initialized optimizer and loss function

# Training function for the Fusion Model
def train_fusion_model(model, data_loader, optimizer, loss_fn, num_epochs=10):
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch in data_loader:
            text_data, image_data, audio_data, labels = batch  # Get batch data
            
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            output = model(text_data, image_data, audio_data)
            
            # Compute loss
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')  # Print loss for each epoch

# Data loader
def data_loader(batch_size, num_batches, text_input_dim, image_input_dim, audio_input_dim):
    for _ in range(num_batches):
        text_data, image_data, audio_data = generate_synthetic_data(batch_size, text_input_dim, image_input_dim, audio_input_dim)
        labels = torch.randint(0, 2, (batch_size, 1)).float()  # Random binary labels
        yield text_data, image_data, audio_data, labels  # Yield batch data

# Main function to run the training process
def main():
    # Define dimensions
    text_input_dim = 512
    image_input_dim = 2048
    audio_input_dim = 128
    fusion_output_dim = 256
    batch_size = 32
    num_batches = 100
    
    # Initialize the Fusion Model
    model = FusionModel(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim)
    
    # Initialize optimizer and loss function
    optimizer, loss_fn = initialize_optimizer_and_loss(model)
    
    # Create a data loader
    data_loader = data_loader(batch_size, num_batches, text_input_dim, image_input_dim, audio_input_dim)
    
    # Train the model
    train_fusion_model(model, data_loader, optimizer, loss_fn, num_epochs=10)

# Run the main function
if __name__ == "__main__":
    main()

# Function to save model checkpoints during training
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved at epoch {epoch}, Loss: {loss:.4f}')

# Function to load model checkpoints for resuming training
def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Checkpoint loaded from epoch {epoch}, Loss: {loss:.4f}')
    return epoch, loss

# Function to evaluate the model's performance on validation data
def evaluate_fusion_model(model, data_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for batch in data_loader:
            text_data, image_data, audio_data, labels = batch
            
            # Forward pass
            output = model(text_data, image_data, audio_data)
            
            # Compute loss
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = (output > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total * 100.0
    print(f'Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy

# Data loader
def validation_loader(batch_size, num_batches, text_input_dim, image_input_dim, audio_input_dim):
    for _ in range(num_batches):
        text_data, image_data, audio_data = generate_synthetic_data(batch_size, text_input_dim, image_input_dim, audio_input_dim)
        labels = torch.randint(0, 2, (batch_size, 1)).float()  # Random binary labels
        yield text_data, image_data, audio_data, labels  # Yield batch data

# Function to train and evaluate the model, including checkpoint saving
def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=10, checkpoint_path='model_checkpoint.pth'):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        
        for batch in train_loader:
            text_data, image_data, audio_data, labels = batch
            
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            output = model(text_data, image_data, audio_data)
            
            # Compute loss
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
        
        # Evaluate the model on validation data
        val_loss, val_accuracy = evaluate_fusion_model(model, val_loader, loss_fn)
        
        # Save the model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

# Function to create a data loader for training and validation
def create_data_loaders(batch_size, train_batches, val_batches, text_input_dim, image_input_dim, audio_input_dim):
    train_loader = data_loader(batch_size, train_batches, text_input_dim, image_input_dim, audio_input_dim)
    val_loader = validation_loader(batch_size, val_batches, text_input_dim, image_input_dim, audio_input_dim)
    return train_loader, val_loader

# Function to run the entire training and evaluation process
def run_training_process():
    # Define dimensions
    text_input_dim = 512
    image_input_dim = 2048
    audio_input_dim = 128
    fusion_output_dim = 256
    batch_size = 32
    train_batches = 100
    val_batches = 20
    num_epochs = 10
    
    # Initialize the Fusion Model
    model = FusionModel(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim)
    
    # Initialize optimizer and loss function
    optimizer, loss_fn = initialize_optimizer_and_loss(model)
    
    # Create data loaders for training and validation
    train_loader, val_loader = create_data_loaders(batch_size, train_batches, val_batches, text_input_dim, image_input_dim, audio_input_dim)
    
    # Train and evaluate the model, saving checkpoints
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=num_epochs)

# Run the training process
if __name__ == "__main__":
    run_training_process()

# Function to load a pretrained model and run inference
def run_inference_on_pretrained_model(filepath, text_features, image_features, audio_features):
    # Define input dimensions
    text_input_dim = text_features.shape[1]
    image_input_dim = image_features.shape[1]
    audio_input_dim = audio_features.shape[1]
    fusion_output_dim = 256
    
    # Initialize the Fusion Model
    model = FusionModel(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim)
    
    # Load the pretrained model from checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer for loading
    load_checkpoint(model, optimizer, filepath)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Run the model on the input features
    with torch.no_grad():
        output = model(text_features, image_features, audio_features)
    
    # Return the model output
    return output

# Usage of inference with a pretrained model
def run_inference():
    # Generate random data for inference
    text_data, image_data, audio_data = generate_synthetic_data(1, 512, 2048, 128)
    
    # Run inference on the pretrained model
    output = run_inference_on_pretrained_model('model_checkpoint.pth', text_data, image_data, audio_data)
    
    # Print the model output
    print("Inference output:", output)

# Call the function to run the inference
run_inference()

# Function to calculate metrics for model evaluation
def calculate_metrics(output, labels):
    # Convert model output to binary predictions
    predicted = (output > 0.5).float()
    
    # Calculate the number of correct predictions
    correct = (predicted == labels).sum().item()
    
    # Calculate accuracy
    accuracy = correct / labels.size(0) * 100.0
    
    # Calculate true positives, false positives, true negatives, false negatives
    true_positive = ((predicted == 1) & (labels == 1)).sum().item()
    false_positive = ((predicted == 1) & (labels == 0)).sum().item()
    true_negative = ((predicted == 0) & (labels == 0)).sum().item()
    false_negative = ((predicted == 0) & (labels == 1)).sum().item()
    
    # Precision, recall, F1-score
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return accuracy, precision, recall, f1_score

# Function to evaluate the model with additional metrics
def evaluate_model_with_metrics(model, data_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1_score = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            text_data, image_data, audio_data, labels = batch
            
            # Forward pass
            output = model(text_data, image_data, audio_data)
            
            # Compute loss
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            
            # Calculate metrics
            accuracy, precision, recall, f1_score = calculate_metrics(output, labels)
            total_accuracy += accuracy * labels.size(0)
            total_precision += precision * labels.size(0)
            total_recall += recall * labels.size(0)
            total_f1_score += f1_score * labels.size(0)
            total_samples += labels.size(0)
    
    # Compute average metrics
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / total_samples
    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples
    avg_f1_score = total_f1_score / total_samples
    
    print(f'Evaluation Metrics - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1 Score: {avg_f1_score:.2f}')
    
    return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score

# Advanced fusion strategy using attention mechanism
class AttentionFusion(nn.Module):
    def __init__(self, fusion_output_dim):
        super(AttentionFusion, self).__init__()
        # Learnable attention weights for each modality
        self.text_attention = nn.Linear(fusion_output_dim, 1)
        self.image_attention = nn.Linear(fusion_output_dim, 1)
        self.audio_attention = nn.Linear(fusion_output_dim, 1)
        
        # Final fusion fully connected layer
        self.fusion_fc = nn.Linear(fusion_output_dim * 3, fusion_output_dim)
        self.relu = nn.ReLU()

    def forward(self, text_encoded, image_encoded, audio_encoded):
        # Calculate attention weights for each modality
        text_weight = torch.sigmoid(self.text_attention(text_encoded))
        image_weight = torch.sigmoid(self.image_attention(image_encoded))
        audio_weight = torch.sigmoid(self.audio_attention(audio_encoded))
        
        # Weighted sum of modalities
        text_weighted = text_weight * text_encoded
        image_weighted = image_weight * image_encoded
        audio_weighted = audio_weight * audio_encoded
        
        # Concatenate the weighted modalities
        fused_features = torch.cat((text_weighted, image_weighted, audio_weighted), dim=1)
        
        # Apply fusion fully connected layer
        fused_output = self.fusion_fc(fused_features)
        fused_output = self.relu(fused_output)
        
        return fused_output

# Updated Fusion Model class to use AttentionFusion
class AttentionFusionModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim):
        super(AttentionFusionModel, self).__init__()
        
        # Encoders for each modality
        self.text_encoder = TextEncoder(text_input_dim, fusion_output_dim)
        self.image_encoder = ImageEncoder(image_input_dim, fusion_output_dim)
        self.audio_encoder = AudioEncoder(audio_input_dim, fusion_output_dim)
        
        # Attention-based fusion module
        self.attention_fusion = AttentionFusion(fusion_output_dim)
        
        # Output layer
        self.output_fc = nn.Linear(fusion_output_dim, 1)

    def forward(self, text_features, image_features, audio_features):
        # Encode each modality
        text_encoded = self.text_encoder(text_features)
        image_encoded = self.image_encoder(image_features)
        audio_encoded = self.audio_encoder(audio_features)
        
        # Apply attention-based fusion
        fused_output = self.attention_fusion(text_encoded, image_encoded, audio_encoded)
        
        # Produce the final output
        output = self.output_fc(fused_output)
        
        return output

# Function to run training and evaluation with the attention-based fusion model
def run_attention_fusion_training():
    # Define input dimensions
    text_input_dim = 512
    image_input_dim = 2048
    audio_input_dim = 128
    fusion_output_dim = 256
    batch_size = 32
    train_batches = 100
    val_batches = 20
    num_epochs = 10
    
    # Initialize the Attention Fusion Model
    model = AttentionFusionModel(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim)
    
    # Initialize optimizer and loss function
    optimizer, loss_fn = initialize_optimizer_and_loss(model)
    
    # Create data loaders for training and validation
    train_loader, val_loader = create_data_loaders(batch_size, train_batches, val_batches, text_input_dim, image_input_dim, audio_input_dim)
    
    # Train and evaluate the model, saving checkpoints
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=num_epochs, checkpoint_path='attention_fusion_checkpoint.pth')

# Run the attention-based fusion model training process
if __name__ == "__main__":
    run_attention_fusion_training()

# Function to run inference with attention-based fusion
def run_attention_fusion_inference(filepath, text_features, image_features, audio_features):
    # Define input dimensions
    text_input_dim = text_features.shape[1]
    image_input_dim = image_features.shape[1]
    audio_input_dim = audio_features.shape[1]
    fusion_output_dim = 256
    
    # Initialize the Attention Fusion Model
    model = AttentionFusionModel(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim)
    
    # Load the pretrained model from checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer for loading
    load_checkpoint(model, optimizer, filepath)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Run the model on the input features
    with torch.no_grad():
        output = model(text_features, image_features, audio_features)
    
    # Return the model output
    return output

# Inference with the attention-based fusion model
def run_attention_fusion_inference():
    # Generate random data for inference
    text_data, image_data, audio_data = generate_synthetic_data(1, 512, 2048, 128)
    
    # Run inference on the pretrained attention-based fusion model
    output = run_attention_fusion_inference('attention_fusion_checkpoint.pth', text_data, image_data, audio_data)
    
    # Print the model output
    print("Attention Fusion Inference output:", output)

# Call the function to run the inference
run_attention_fusion_inference()

# Function to log training metrics during training
def log_training_metrics(epoch, train_loss, val_loss, val_accuracy, val_precision, val_recall, val_f1_score):
    print(f"Epoch {epoch + 1}:")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1 Score: {val_f1_score:.4f}")

# Function to update learning rate during training using a learning rate scheduler
def update_learning_rate(optimizer, epoch, initial_lr, lr_decay_factor, decay_every):
    # Check if learning rate needs to be updated based on epoch count
    if (epoch + 1) % decay_every == 0:
        new_lr = initial_lr * (lr_decay_factor ** ((epoch + 1) // decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Learning rate updated to {new_lr:.6f}")

# Function to perform model training with learning rate scheduling and logging
def train_model_with_logging(model, train_loader, val_loader, optimizer, loss_fn, initial_lr, lr_decay_factor=0.5, decay_every=5, num_epochs=10, checkpoint_path='model_with_logging_checkpoint.pth'):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        
        # Training loop
        for batch in train_loader:
            text_data, image_data, audio_data, labels = batch
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            output = model(text_data, image_data, audio_data)
            
            # Compute loss
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate the model on validation data
        val_loss, val_accuracy, val_precision, val_recall, val_f1_score = evaluate_model_with_metrics(model, val_loader, loss_fn)
        
        # Log metrics for the current epoch
        log_training_metrics(epoch, avg_train_loss, val_loss, val_accuracy, val_precision, val_recall, val_f1_score)
        
        # Update learning rate
        update_learning_rate(optimizer, epoch, initial_lr, lr_decay_factor, decay_every)
        
        # Save the model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

# Advanced Feature Extraction for Multimodal Inputs

class AdvancedTextEncoder(nn.Module):
    def __init__(self, text_input_dim, text_hidden_dim, text_layers=2):
        super(AdvancedTextEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(text_input_dim if i == 0 else text_hidden_dim, text_hidden_dim) for i in range(text_layers)])
        self.relu = nn.ReLU()
    
    def forward(self, text_features):
        x = text_features
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x

class AdvancedImageEncoder(nn.Module):
    def __init__(self, image_input_dim, image_hidden_dim, image_layers=2):
        super(AdvancedImageEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(image_input_dim if i == 0 else image_hidden_dim, image_hidden_dim) for i in range(image_layers)])
        self.relu = nn.ReLU()
    
    def forward(self, image_features):
        x = image_features
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x

class AdvancedAudioEncoder(nn.Module):
    def __init__(self, audio_input_dim, audio_hidden_dim, audio_layers=2):
        super(AdvancedAudioEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(audio_input_dim if i == 0 else audio_hidden_dim, audio_hidden_dim) for i in range(audio_layers)])
        self.relu = nn.ReLU()
    
    def forward(self, audio_features):
        x = audio_features
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x

# Advanced Fusion Model using the new encoders
class AdvancedFusionModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim, text_layers=2, image_layers=2, audio_layers=2):
        super(AdvancedFusionModel, self).__init__()
        
        # Advanced encoders for each modality
        self.text_encoder = AdvancedTextEncoder(text_input_dim, fusion_output_dim, text_layers)
        self.image_encoder = AdvancedImageEncoder(image_input_dim, fusion_output_dim, image_layers)
        self.audio_encoder = AdvancedAudioEncoder(audio_input_dim, fusion_output_dim, audio_layers)
        
        # Attention-based fusion module
        self.attention_fusion = AttentionFusion(fusion_output_dim)
        
        # Output layer
        self.output_fc = nn.Linear(fusion_output_dim, 1)

    def forward(self, text_features, image_features, audio_features):
        # Encode each modality
        text_encoded = self.text_encoder(text_features)
        image_encoded = self.image_encoder(image_features)
        audio_encoded = self.audio_encoder(audio_features)
        
        # Apply attention-based fusion
        fused_output = self.attention_fusion(text_encoded, image_encoded, audio_encoded)
        
        # Produce the final output
        output = self.output_fc(fused_output)
        
        return output

# Function to run advanced training with advanced feature extraction and logging
def run_advanced_training():
    # Define input dimensions
    text_input_dim = 512
    image_input_dim = 2048
    audio_input_dim = 128
    fusion_output_dim = 256
    batch_size = 32
    train_batches = 100
    val_batches = 20
    num_epochs = 10
    initial_lr = 0.001
    
    # Initialize the Advanced Fusion Model
    model = AdvancedFusionModel(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim, text_layers=3, image_layers=3, audio_layers=3)
    
    # Initialize optimizer and loss function
    optimizer, loss_fn = initialize_optimizer_and_loss(model)
    
    # Create data loaders for training and validation
    train_loader, val_loader = create_data_loaders(batch_size, train_batches, val_batches, text_input_dim, image_input_dim, audio_input_dim)
    
    # Train and evaluate the model with learning rate scheduling and logging
    train_model_with_logging(model, train_loader, val_loader, optimizer, loss_fn, initial_lr, num_epochs=num_epochs, checkpoint_path='advanced_fusion_checkpoint.pth')

# Run the advanced training process
if __name__ == "__main__":
    run_advanced_training()

# Function to run advanced inference with pretrained model
def run_advanced_inference(filepath, text_features, image_features, audio_features):
    # Define input dimensions
    text_input_dim = text_features.shape[1]
    image_input_dim = image_features.shape[1]
    audio_input_dim = audio_features.shape[1]
    fusion_output_dim = 256
    
    # Initialize the Advanced Fusion Model
    model = AdvancedFusionModel(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim, text_layers=3, image_layers=3, audio_layers=3)
    
    # Load the pretrained model from checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer for loading
    load_checkpoint(model, optimizer, filepath)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Run the model on the input features
    with torch.no_grad():
        output = model(text_features, image_features, audio_features)
    
    # Return the model output
    return output

# Inference using advanced fusion model
def run_advanced_inference():
    # Generate random data for inference
    text_data, image_data, audio_data = generate_synthetic_data(1, 512, 2048, 128)
    
    # Run inference on the pretrained advanced fusion model
    output = run_advanced_inference('advanced_fusion_checkpoint.pth', text_data, image_data, audio_data)
    
    # Print the model output
    print("Advanced Fusion Model Inference output:", output)

# Call the function to run the inference
run_advanced_inference()

# Function to visualize attention weights during inference
def visualize_attention_weights(text_weight, image_weight, audio_weight):
    import matplotlib.pyplot as plt
    
    # Create a bar plot for attention weights
    labels = ['Text', 'Image', 'Audio']
    weights = [text_weight.item(), image_weight.item(), audio_weight.item()]
    
    plt.bar(labels, weights, color=['blue', 'green', 'red'])
    plt.title('Attention Weights Across Modalities')
    plt.ylabel('Weight')
    plt.show()

# Modified AttentionFusion class with weight visualization during inference
class AttentionFusionWithVisualization(nn.Module):
    def __init__(self, fusion_output_dim):
        super(AttentionFusionWithVisualization, self).__init__()
        # Learnable attention weights for each modality
        self.text_attention = nn.Linear(fusion_output_dim, 1)
        self.image_attention = nn.Linear(fusion_output_dim, 1)
        self.audio_attention = nn.Linear(fusion_output_dim, 1)
        
        # Final fusion fully connected layer
        self.fusion_fc = nn.Linear(fusion_output_dim * 3, fusion_output_dim)
        self.relu = nn.ReLU()

    def forward(self, text_encoded, image_encoded, audio_encoded, visualize=False):
        # Calculate attention weights for each modality
        text_weight = torch.sigmoid(self.text_attention(text_encoded))
        image_weight = torch.sigmoid(self.image_attention(image_encoded))
        audio_weight = torch.sigmoid(self.audio_attention(audio_encoded))
        
        # Weighted sum of modalities
        text_weighted = text_weight * text_encoded
        image_weighted = image_weight * image_encoded
        audio_weighted = audio_weight * audio_encoded
        
        # Concatenate the weighted modalities
        fused_features = torch.cat((text_weighted, image_weighted, audio_weighted), dim=1)
        
        # Apply fusion fully connected layer
        fused_output = self.fusion_fc(fused_features)
        fused_output = self.relu(fused_output)
        
        # Visualize attention weights if the flag is set
        if visualize:
            visualize_attention_weights(text_weight, image_weight, audio_weight)
        
        return fused_output

# Advanced Fusion Model with Attention Visualization
class AdvancedFusionModelWithVisualization(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim, text_layers=2, image_layers=2, audio_layers=2):
        super(AdvancedFusionModelWithVisualization, self).__init__()
        
        # Advanced encoders for each modality
        self.text_encoder = AdvancedTextEncoder(text_input_dim, fusion_output_dim, text_layers)
        self.image_encoder = AdvancedImageEncoder(image_input_dim, fusion_output_dim, image_layers)
        self.audio_encoder = AdvancedAudioEncoder(audio_input_dim, fusion_output_dim, audio_layers)
        
        # Attention-based fusion module with visualization
        self.attention_fusion = AttentionFusionWithVisualization(fusion_output_dim)
        
        # Output layer
        self.output_fc = nn.Linear(fusion_output_dim, 1)

    def forward(self, text_features, image_features, audio_features, visualize=False):
        # Encode each modality
        text_encoded = self.text_encoder(text_features)
        image_encoded = self.image_encoder(image_features)
        audio_encoded = self.audio_encoder(audio_features)
        
        # Apply attention-based fusion with visualization
        fused_output = self.attention_fusion(text_encoded, image_encoded, audio_encoded, visualize)
        
        # Produce the final output
        output = self.output_fc(fused_output)
        
        return output

# Function to run inference with visualization
def run_inference_with_visualization(filepath, text_features, image_features, audio_features, visualize=True):
    # Define input dimensions
    text_input_dim = text_features.shape[1]
    image_input_dim = image_features.shape[1]
    audio_input_dim = audio_features.shape[1]
    fusion_output_dim = 256
    
    # Initialize the Advanced Fusion Model with Visualization
    model = AdvancedFusionModelWithVisualization(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim, text_layers=3, image_layers=3, audio_layers=3)
    
    # Load the pretrained model from checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer for loading
    load_checkpoint(model, optimizer, filepath)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Run the model on the input features with visualization
    with torch.no_grad():
        output = model(text_features, image_features, audio_features, visualize)
    
    return output

# Inference with attention visualization
def run_inference_with_visualization():
    # Generate random data for inference
    text_data, image_data, audio_data = generate_synthetic_data(1, 512, 2048, 128)
    
    # Run inference on the pretrained model with attention visualization
    output = run_inference_with_visualization('advanced_fusion_checkpoint.pth', text_data, image_data, audio_data, visualize=True)
    
    # Print the model output
    print("Advanced Fusion Model with Visualization Inference output:", output)

# Call the function to run the inference with visualization
run_inference_with_visualization()

# Function to fine-tune the model using transfer learning
def fine_tune_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=5, checkpoint_path='fine_tuned_checkpoint.pth'):
    best_val_loss = float('inf')
    
    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        
        # Training loop
        for batch in train_loader:
            text_data, image_data, audio_data, labels = batch
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            output = model(text_data, image_data, audio_data)
            
            # Compute loss
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate the model on validation data
        val_loss, val_accuracy, val_precision, val_recall, val_f1_score = evaluate_model_with_metrics(model, val_loader, loss_fn)
        
        # Log metrics for the current epoch
        log_training_metrics(epoch, avg_train_loss, val_loss, val_accuracy, val_precision, val_recall, val_f1_score)
        
        # Save the model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

# Function to run fine-tuning process
def run_fine_tuning():
    # Define input dimensions
    text_input_dim = 512
    image_input_dim = 2048
    audio_input_dim = 128
    fusion_output_dim = 256
    batch_size = 32
    train_batches = 50
    val_batches = 10
    num_epochs = 5
    
    # Initialize the Advanced Fusion Model with Visualization
    model = AdvancedFusionModelWithVisualization(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim, text_layers=3, image_layers=3, audio_layers=3)
    
    # Initialize optimizer and loss function
    optimizer, loss_fn = initialize_optimizer_and_loss(model)
    
    # Create data loaders for training and validation
    train_loader, val_loader = create_data_loaders(batch_size, train_batches, val_batches, text_input_dim, image_input_dim, audio_input_dim)
    
    # Fine-tune the model
    fine_tune_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=num_epochs, checkpoint_path='fine_tuned_checkpoint.pth')

# Run the fine-tuning process
if __name__ == "__main__":
    run_fine_tuning()

# Function to test performance after fine-tuning
def test_fine_tuned_model(filepath, test_loader):
    # Define input dimensions for the test model
    text_input_dim = 512
    image_input_dim = 2048
    audio_input_dim = 128
    fusion_output_dim = 256
    
    # Initialize the model with visualization for testing
    model = AdvancedFusionModelWithVisualization(text_input_dim, image_input_dim, audio_input_dim, fusion_output_dim, text_layers=3, image_layers=3, audio_layers=3)
    
    # Load the fine-tuned model from checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer for loading
    load_checkpoint(model, optimizer, filepath)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Testing loop
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            text_data, image_data, audio_data, labels = batch
            
            # Forward pass
            output = model(text_data, image_data, audio_data)
            
            # Compute loss and accuracy
            loss = nn.BCEWithLogitsLoss()(output, labels)
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total * 100.0
    
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

# Function to test fine-tuned model performance
def run_test_fine_tuned_model():
    # Create test loader
    test_loader = data_loader(batch_size=32, num_batches=20, text_input_dim=512, image_input_dim=2048, audio_input_dim=128)
    
    # Test the fine-tuned model performance
    test_fine_tuned_model('fine_tuned_checkpoint.pth', test_loader)

# Run the test on fine-tuned model
run_test_fine_tuned_model()