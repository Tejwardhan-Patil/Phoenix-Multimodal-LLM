import argparse
import yaml
import os
from models import text, image, audio, multimodal
from utils import data_loader, metrics
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_experiment(config):
    logging.info(f"Setting up experiment for modality: {config['experiment']['modality']}")
    
    if config['experiment']['modality'] == 'text':
        model = text.gpt_model.GPTModel(config['model'])
    elif config['experiment']['modality'] == 'image':
        model = image.cnn_model.CNNModel(config['model'])
    elif config['experiment']['modality'] == 'audio':
        model = audio.rnn_audio_model.RNNModel(config['model'])
    elif config['experiment']['modality'] == 'multimodal':
        model = multimodal.fusion_model.FusionModel(config['model'])
    else:
        raise ValueError(f"Unknown modality: {config['experiment']['modality']}")
    
    logging.info(f"Model for {config['experiment']['modality']} loaded successfully.")
    return model

def select_optimizer(model, config):
    logging.info(f"Selecting optimizer: {config['training']['optimizer']}")
    
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    logging.info("Optimizer initialized.")
    return optimizer

def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    logging.info(f"Saving checkpoint for epoch {epoch}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    logging.info(f"Loading checkpoint from {path}")
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logging.info(f"Checkpoint loaded. Resuming from epoch {epoch}")
        return epoch
    else:
        logging.warning(f"No checkpoint found at {path}. Starting from scratch.")
        return 0

def run_experiment(config):
    # Load dataset
    logging.info(f"Loading dataset from {config['data']['path']}")
    data = data_loader.load_data(config['data'])
    
    # Load model
    model = setup_experiment(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and scheduler
    optimizer = select_optimizer(model, config)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_step'], gamma=0.1)
    
    # Early stopping params
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Load checkpoint
    start_epoch = load_checkpoint(model, optimizer) if config['experiment']['resume'] else 0

    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        logging.info(f"Starting epoch {epoch + 1}")
        for batch_idx, batch in enumerate(data['train_loader']):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % config['experiment']['log_interval'] == 0:
                logging.info(f"Train Epoch: {epoch + 1} [{batch_idx * len(inputs)}/{len(data['train_loader'].dataset)} "
                             f"({100. * batch_idx / len(data['train_loader']):.0f}%)]\tLoss: {loss.item():.6f}")
        
        # Adjust learning rate
        scheduler.step()
        
        # Validation after each epoch
        val_loss, val_acc = evaluate(model, data['val_loader'], device, config)
        
        # Checkpoint and early stopping
        if val_loss < best_val_loss:
            save_checkpoint(model, optimizer, epoch)
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['experiment']['patience']:
            logging.info("Early stopping triggered.")
            break

def evaluate(model, val_loader, device, config):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    logging.info(f"Validation Loss: {val_loss / len(val_loader):.6f}, Accuracy: {accuracy:.2f}%, "
                 f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    
    # Custom multimodal metrics
    if config['experiment']['modality'] == 'multimodal':
        multimodal_score = metrics.calculate_multimodal_score(model, val_loader, device)
        logging.info(f"Multimodal Alignment Score: {multimodal_score}")
    
    return val_loss / len(val_loader), accuracy

def main():
    parser = argparse.ArgumentParser(description="Run Experiment")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = load_config(config_path)
    run_experiment(config)

if __name__ == "__main__":
    main()