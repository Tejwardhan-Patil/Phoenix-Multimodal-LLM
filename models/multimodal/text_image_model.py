import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from torchvision import models, transforms
import torch.nn.functional as F

# Text Encoder Module
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', embedding_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
        cls_output = self.dropout(cls_output)
        embeddings = self.fc(cls_output)
        return embeddings

# Image Encoder Module
class ImageEncoder(nn.Module):
    def __init__(self, pretrained_model='resnet50', embedding_dim=512):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final fully connected layer
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)  # Flatten the features
        features = self.dropout(features)
        embeddings = self.fc(features)
        return embeddings

# Multimodal Model Class
class MultimodalModel(nn.Module):
    def __init__(self, text_embedding_dim=512, image_embedding_dim=512):
        super(MultimodalModel, self).__init__()
        self.text_encoder = TextEncoder(embedding_dim=text_embedding_dim)
        self.image_encoder = ImageEncoder(embedding_dim=image_embedding_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        self.temperature.requires_grad = True  # Temperature parameter is learnable

    def forward(self, input_ids, attention_mask, images):
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        image_embeddings = self.image_encoder(images)

        # Normalize the embeddings for stability
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        image_embeddings = F.normalize(image_embeddings, dim=-1)

        # Calculate similarity scores (dot product)
        logits_per_text = torch.matmul(text_embeddings, image_embeddings.t()) / self.temperature
        logits_per_image = logits_per_text.t()

        return logits_per_text, logits_per_image

    def get_text_embeddings(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)

    def get_image_embeddings(self, images):
        return self.image_encoder(images)

# Function to calculate contrastive loss (Cross Entropy Loss)
def contrastive_loss(logits_per_text, logits_per_image):
    labels = torch.arange(logits_per_text.size(0)).to(logits_per_text.device)
    loss_text = nn.CrossEntropyLoss()(logits_per_text, labels)
    loss_image = nn.CrossEntropyLoss()(logits_per_image, labels)
    return (loss_text + loss_image) / 2

# Preprocessing transformations for images
def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Optimizer and Scheduler configuration
def configure_optimizers(model, learning_rate=1e-4, weight_decay=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    return optimizer, scheduler

# Forward pass during training
def training_step(model, batch, device):
    input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
    input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

    logits_per_text, logits_per_image = model(input_ids, attention_mask, images)
    loss = contrastive_loss(logits_per_text, logits_per_image)

    return loss

# Backward pass and optimizer step
def backward_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Sample dataset for demonstration
class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, text_inputs, image_inputs, tokenizer, transform):
        self.text_inputs = text_inputs
        self.image_inputs = image_inputs
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.text_inputs)

    def __getitem__(self, idx):
        text = self.text_inputs[idx]
        image = self.image_inputs[idx]

        # Tokenize the text
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Apply image transformations
        image = self.transform(image)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': image
        }

# Training loop
def train_model(model, dataloader, optimizer, scheduler, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_loss = training_step(model, batch, device)
            backward_step(optimizer, batch_loss)
            epoch_loss += batch_loss.item()

        # Step the learning rate scheduler
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation step
def validation_step(model, batch, device):
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
        input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

        logits_per_text, logits_per_image = model(input_ids, attention_mask, images)
        loss = contrastive_loss(logits_per_text, logits_per_image)

    return loss.item()

# Full validation loop for an epoch
def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_loss = validation_step(model, batch, device)
            total_loss += batch_loss

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# Save model checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, path)
    print(f"Checkpoint saved at epoch {epoch}")

# Load model checkpoint
def load_checkpoint(model, optimizer, scheduler, path='checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded, resuming from epoch {epoch} with loss {loss}")
    return epoch, loss

# Learning rate scheduler adjustment
def adjust_learning_rate(optimizer, epoch, initial_lr=1e-4, decay_rate=0.1, decay_epochs=10):
    """Sets the learning rate to the initial LR decayed by 10 every decay_epochs."""
    lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Learning rate adjusted to {lr}")

# Function to predict with the model
def multimodal_predict(model, input_ids, attention_mask, images, device):
    model.eval()
    input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)
    with torch.no_grad():
        logits_per_text, logits_per_image = model(input_ids, attention_mask, images)
        # Retrieving predictions (based on highest similarity)
        predictions = torch.argmax(logits_per_text, dim=1)
    return predictions

# Advanced Data Augmentation for Images
def advanced_image_augmentations():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to visualize predictions and corresponding images
import matplotlib.pyplot as plt

def visualize_predictions(images, predictions, class_names):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, image in enumerate(images):
        img = image.permute(1, 2, 0).cpu().numpy()
        img = img * 0.229 + 0.485  # Denormalize
        axs[i].imshow(img)
        axs[i].set_title(class_names[predictions[i]])
        axs[i].axis('off')
    plt.show()

# Mixed Precision Training (for faster training with less memory usage)
from torch.cuda.amp import autocast, GradScaler

def mixed_precision_training_step(model, batch, optimizer, scaler, device):
    input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
    input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

    with autocast():
        logits_per_text, logits_per_image = model(input_ids, attention_mask, images)
        loss = contrastive_loss(logits_per_text, logits_per_image)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

# Full training loop with mixed precision
def train_with_mixed_precision(model, dataloader, optimizer, scheduler, device, num_epochs=10):
    model.train()
    scaler = GradScaler()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_loss = mixed_precision_training_step(model, batch, optimizer, scaler, device)
            epoch_loss += batch_loss

        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Function to compute the accuracy of predictions
def compute_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# Full evaluation loop with accuracy calculation
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    total_accuracy = 0.0
    total_loss = 0.0
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, images, labels = batch['input_ids'], batch['attention_mask'], batch['images'], batch['labels']
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

            logits_per_text, logits_per_image = model(input_ids, attention_mask, images)
            loss = contrastive_loss(logits_per_text, logits_per_image)
            total_loss += loss.item()

            predictions = torch.argmax(logits_per_text, dim=1)
            accuracy = compute_accuracy(predictions, labels)
            total_accuracy += accuracy

    avg_loss = total_loss / total_batches
    avg_accuracy = total_accuracy / total_batches
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")
    return avg_loss, avg_accuracy

# Learning rate finder to help determine optimal learning rates
def find_learning_rate(model, dataloader, optimizer, device, init_value=1e-8, final_value=10.0, beta=0.98):
    model.train()
    num_batches = len(dataloader) - 1
    mult = (final_value / init_value) ** (1/num_batches)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.0
    best_loss = 0.0
    losses = []
    log_lrs = []

    for batch_num, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
        input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

        logits_per_text, logits_per_image = model(input_ids, attention_mask, images)
        loss = contrastive_loss(logits_per_text, logits_per_image)

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_num + 1))

        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(torch.log10(torch.tensor(lr)))

        loss.backward()
        optimizer.step()

        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses

# Cosine similarity metric for multimodal alignment
def cosine_similarity(a, b):
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return torch.mm(a_norm, b_norm.t())

# Multimodal alignment score
def multimodal_alignment_score(model, dataloader, device):
    model.eval()
    total_alignment = 0.0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
            input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

            text_embeddings = model.get_text_embeddings(input_ids, attention_mask)
            image_embeddings = model.get_image_embeddings(images)

            similarity_matrix = cosine_similarity(text_embeddings, image_embeddings)
            alignment_score = torch.mean(similarity_matrix.diag())  # Diagonal elements indicate matching pairs
            total_alignment += alignment_score.item()

    avg_alignment_score = total_alignment / total_batches
    print(f"Multimodal Alignment Score: {avg_alignment_score:.4f}")
    return avg_alignment_score

# Fine-tuning the model with a different task
def fine_tune_model(model, dataloader, optimizer, scheduler, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids, attention_mask, images, labels = batch['input_ids'], batch['attention_mask'], batch['images'], batch['labels']
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits_per_text, logits_per_image = model(input_ids, attention_mask, images)

            # Fine-tuning task that aligns with the original multimodal contrastive task
            loss = contrastive_loss(logits_per_text, logits_per_image)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Fine-Tuning Loss: {epoch_loss:.4f}")

# Batch-wise accuracy computation during training
def batch_accuracy(predictions, labels):
    correct = predictions.eq(labels.view_as(predictions)).sum().item()
    return correct / len(labels)

# Function to load image and text features for multimodal evaluation
def load_multimodal_features(text_data, image_data, tokenizer, transform):
    text_features = []
    image_features = []
    
    for text, image in zip(text_data, image_data):
        # Tokenize the text
        encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        text_features.append((input_ids, attention_mask))

        # Transform the image
        image = transform(image)
        image_features.append(image)

    return text_features, image_features

# Class to handle multimodal data preparation for evaluation
class MultimodalDataLoader(torch.utils.data.Dataset):
    def __init__(self, text_features, image_features):
        self.text_features = text_features
        self.image_features = image_features

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.text_features[idx]
        image = self.image_features[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': image
        }

# Function to extract and cache multimodal embeddings for fast inference
def cache_multimodal_embeddings(model, dataloader, device, cache_dir='cache/'):
    model.eval()
    text_embeddings_cache = []
    image_embeddings_cache = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
            input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

            # Extract text and image embeddings
            text_embeddings = model.get_text_embeddings(input_ids, attention_mask)
            image_embeddings = model.get_image_embeddings(images)

            # Cache the embeddings
            text_embeddings_cache.append(text_embeddings.cpu())
            image_embeddings_cache.append(image_embeddings.cpu())

    # Save cached embeddings
    torch.save(torch.cat(text_embeddings_cache), f'{cache_dir}/text_embeddings.pt')
    torch.save(torch.cat(image_embeddings_cache), f'{cache_dir}/image_embeddings.pt')
    print(f"Cached embeddings saved to {cache_dir}")

# Function to load cached embeddings for evaluation
def load_cached_embeddings(cache_dir='cache/'):
    text_embeddings = torch.load(f'{cache_dir}/text_embeddings.pt')
    image_embeddings = torch.load(f'{cache_dir}/image_embeddings.pt')
    print(f"Loaded cached embeddings from {cache_dir}")
    return text_embeddings, image_embeddings

# Nearest neighbor search for fast retrieval of similar multimodal embeddings
def nearest_neighbor_search(query_embedding, embeddings, top_k=5):
    # Compute cosine similarity between the query and the cached embeddings
    similarity_scores = cosine_similarity(query_embedding, embeddings)
    top_scores, top_indices = torch.topk(similarity_scores, top_k, dim=1)
    return top_indices, top_scores

# Inference with nearest neighbor search for text-image retrieval
def multimodal_retrieval(model, input_ids, attention_mask, images, text_embeddings, image_embeddings, device, top_k=5):
    model.eval()
    input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

    with torch.no_grad():
        # Get current batch embeddings
        text_embedding = model.get_text_embeddings(input_ids, attention_mask)
        image_embedding = model.get_image_embeddings(images)

        # Perform nearest neighbor search on both text and image embeddings
        text_nn_indices, text_nn_scores = nearest_neighbor_search(text_embedding, image_embeddings, top_k=top_k)
        image_nn_indices, image_nn_scores = nearest_neighbor_search(image_embedding, text_embeddings, top_k=top_k)

    return text_nn_indices, text_nn_scores, image_nn_indices, image_nn_scores

# Function for zero-shot learning
def zero_shot_classification(model, input_ids, attention_mask, class_prompts, tokenizer, device):
    model.eval()
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    
    with torch.no_grad():
        text_embeddings = model.get_text_embeddings(input_ids, attention_mask)

        # Encode class prompts
        class_encodings = [tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device) for prompt in class_prompts]
        class_embeddings = [model.get_text_embeddings(encoding['input_ids'], encoding['attention_mask']) for encoding in class_encodings]

        # Stack class embeddings
        class_embeddings = torch.stack(class_embeddings).squeeze()

        # Compute similarity between input text and class embeddings
        similarity_scores = cosine_similarity(text_embeddings, class_embeddings)
        predicted_class = torch.argmax(similarity_scores, dim=1)

    return predicted_class

# Function for logging model metrics during training
def log_metrics(metrics, epoch, logger):
    logger.info(f"Epoch {epoch}:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

# Setup logger for training
import logging

def setup_logger(log_file='training.log'):
    logger = logging.getLogger('multimodal_training')
    logger.setLevel(logging.INFO)

    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# Function for evaluating multiple metrics at once
def evaluate_multiple_metrics(model, dataloader, device, metrics_list):
    model.eval()
    total_metrics = {metric: 0.0 for metric in metrics_list}

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, images, labels = batch['input_ids'], batch['attention_mask'], batch['images'], batch['labels']
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

            logits_per_text, logits_per_image = model(input_ids, attention_mask, images)

            # Compute the metrics
            predictions = torch.argmax(logits_per_text, dim=1)
            accuracy = compute_accuracy(predictions, labels)
            total_metrics['accuracy'] += accuracy

    # Average the metrics
    for key in total_metrics.keys():
        total_metrics[key] /= len(dataloader)

    return total_metrics

# Define a F1-Score function for binary or multiclass classification
def f1_score(predictions, labels, average='macro'):
    from sklearn.metrics import f1_score as sk_f1_score
    return sk_f1_score(labels.cpu(), predictions.cpu(), average=average)

# Function to compute precision
def precision_score(predictions, labels, average='macro'):
    from sklearn.metrics import precision_score as sk_precision_score
    return sk_precision_score(labels.cpu(), predictions.cpu(), average=average)

# Function to compute recall
def recall_score(predictions, labels, average='macro'):
    from sklearn.metrics import recall_score as sk_recall_score
    return sk_recall_score(labels.cpu(), predictions.cpu(), average=average)

# Function to plot training and validation loss curves
def plot_loss_curves(train_losses, val_losses):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

# Function to plot learning rate finder results
def plot_lr_finder(log_lrs, losses):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(log_lrs, losses)
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.title('Learning Rate Finder')
    plt.show()

# Adding a mechanism to save and load model training statistics (loss, accuracy, etc)
def save_training_statistics(train_stats, filename='training_stats.pt'):
    torch.save(train_stats, filename)
    print(f"Training statistics saved to {filename}")

def load_training_statistics(filename='training_stats.pt'):
    train_stats = torch.load(filename)
    print(f"Training statistics loaded from {filename}")
    return train_stats

# Function to calculate and log detailed classification report
def classification_report(predictions, labels, class_names):
    from sklearn.metrics import classification_report as sk_classification_report
    report = sk_classification_report(labels.cpu(), predictions.cpu(), target_names=class_names, zero_division=0)
    print(report)
    return report

# Function to save the classification report to a file
def save_classification_report(report, filename='classification_report.txt'):
    with open(filename, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {filename}")

# Function to implement early stopping based on validation loss
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# DataLoader to handle multimodal data from different sources
def create_dataloader(text_inputs, image_inputs, labels, tokenizer, transform, batch_size=32):
    dataset = MultimodalDataset(text_inputs, image_inputs, tokenizer, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Evaluation function for fine-tuned models
def evaluate_fine_tuned_model(model, dataloader, device, class_names):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, images, labels = batch['input_ids'], batch['attention_mask'], batch['images'], batch['labels']
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

            logits_per_text, logits_per_image = model(input_ids, attention_mask, images)
            loss = contrastive_loss(logits_per_text, logits_per_image)
            total_loss += loss.item()

            # Get predictions from logits
            predictions = torch.argmax(logits_per_text, dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = compute_accuracy(all_predictions, all_labels)
    print(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Generate detailed classification report
    report = classification_report(all_predictions, all_labels, class_names)
    return avg_loss, accuracy, report

# Define a training manager class for encapsulating training, evaluation, and saving checkpoints
class TrainingManager:
    def __init__(self, model, optimizer, scheduler, device, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.best_loss = float('inf')

    def train(self, dataloader, num_epochs, val_dataloader=None, early_stopping=None):
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in dataloader:
                input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
                input_ids, attention_mask, images = input_ids.to(self.device), attention_mask.to(self.device), images.to(self.device)

                self.optimizer.zero_grad()
                logits_per_text, logits_per_image = self.model(input_ids, attention_mask, images)
                loss = contrastive_loss(logits_per_text, logits_per_image)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Adjust learning rate
            self.scheduler.step(epoch_loss)

            # Log metrics for the epoch
            if self.logger:
                log_metrics({'training_loss': epoch_loss / len(dataloader)}, epoch, self.logger)

            # Early stopping and validation
            if val_dataloader:
                val_loss, _ = self.validate(val_dataloader)
                if early_stopping:
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # Save the best model
            if val_dataloader and val_loss < self.best_loss:
                self.best_loss = val_loss
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_loss, path='best_model.pth')

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
                input_ids, attention_mask, images = input_ids.to(self.device), attention_mask.to(self.device), images.to(self.device)

                logits_per_text, logits_per_image = self.model(input_ids, attention_mask, images)
                loss = contrastive_loss(logits_per_text, logits_per_image)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss, self.model

# Multi-GPU support using DataParallel
def enable_multi_gpu(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model

# Function to evaluate with multi-GPU support
def evaluate_with_multi_gpu(model, dataloader, device, class_names):
    model = enable_multi_gpu(model)
    return evaluate_fine_tuned_model(model, dataloader, device, class_names)

# Function to load and prepare data for testing
def load_test_data(text_data, image_data, tokenizer, transform):
    text_features, image_features = load_multimodal_features(text_data, image_data, tokenizer, transform)
    dataset = MultimodalDataLoader(text_features, image_features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader

# Prediction function for loading test data and returning predictions
def predict_on_test_data(model, test_dataloader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, images = batch['input_ids'], batch['attention_mask'], batch['images']
            input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

            logits_per_text, logits_per_image = model(input_ids, attention_mask, images)
            predictions = torch.argmax(logits_per_text, dim=1)
            all_predictions.append(predictions.cpu())

    return torch.cat(all_predictions)

# Function to calculate accuracy for the test set
def test_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    accuracy = correct / len(labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# Function to save model and optimizer state after training
def save_final_model(model, optimizer, path='final_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Final model saved at {path}")