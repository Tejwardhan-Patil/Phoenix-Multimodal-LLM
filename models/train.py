import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.multimodal.fusion_model import FusionModel 
from data_loader import MultimodalDataset  
from utils.metrics import compute_loss, compute_metrics 
import logging
import time

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 20
learning_rate = 1e-4
log_interval = 10
save_model_interval = 5
save_dir = 'models/pretrained/'

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Dataset Loaders
def load_datasets():
    logging.info('Loading datasets...')
    train_dataset = MultimodalDataset(split='train')
    val_dataset = MultimodalDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Initialize Model
def initialize_model():
    logging.info('Initializing model...')
    model = FusionModel().to(device)
    return model

# Loss functions for each modality
def get_criterions():
    text_criterion = nn.CrossEntropyLoss()
    image_criterion = nn.MSELoss()
    audio_criterion = nn.L1Loss()
    return text_criterion, image_criterion, audio_criterion

# Optimizer and Scheduler
def configure_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return optimizer, scheduler

# Training Step
def train_step(model, batch, criterions, optimizer):
    optimizer.zero_grad()

    text_inputs, image_inputs, audio_inputs, targets = batch['text'], batch['image'], batch['audio'], batch['targets']
    text_inputs, image_inputs, audio_inputs, targets = text_inputs.to(device), image_inputs.to(device), audio_inputs.to(device), targets.to(device)

    text_output, image_output, audio_output = model(text_inputs, image_inputs, audio_inputs)

    text_loss = criterions[0](text_output, targets['text'])
    image_loss = criterions[1](image_output, targets['image'])
    audio_loss = criterions[2](audio_output, targets['audio'])

    total_loss = text_loss + image_loss + audio_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()

# Validation Step
def validation_step(model, batch, criterions):
    with torch.no_grad():
        text_inputs, image_inputs, audio_inputs, targets = batch['text'], batch['image'], batch['audio'], batch['targets']
        text_inputs, image_inputs, audio_inputs, targets = text_inputs.to(device), image_inputs.to(device), audio_inputs.to(device), targets.to(device)

        text_output, image_output, audio_output = model(text_inputs, image_inputs, audio_inputs)

        text_loss = criterions[0](text_output, targets['text'])
        image_loss = criterions[1](image_output, targets['image'])
        audio_loss = criterions[2](audio_output, targets['audio'])

        total_loss = text_loss + image_loss + audio_loss

    return total_loss.item()

# Training Loop
def train(model, train_loader, criterions, optimizer, scheduler):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        batch_loss = train_step(model, batch, criterions, optimizer)
        total_loss += batch_loss

        if batch_idx % log_interval == 0:
            logging.info(f'Batch {batch_idx}, Loss: {batch_loss:.4f}')

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    logging.info(f'Epoch finished, Avg Loss: {avg_loss:.4f}, Time taken: {time.time() - start_time:.2f} sec')
    return avg_loss

# Validation Loop
def validate(model, val_loader, criterions):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch_loss = validation_step(model, batch, criterions)
            total_loss += batch_loss

    avg_loss = total_loss / len(val_loader)
    logging.info(f'Validation Avg Loss: {avg_loss:.4f}')
    return avg_loss

# Save Model Checkpoint
def save_model(epoch, model, optimizer, scheduler, loss, save_dir=save_dir):
    save_path = f'{save_dir}model_epoch_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, save_path)
    logging.info(f'Model checkpoint saved at {save_path}')

# Main Training Loop
def main():
    train_loader, val_loader = load_datasets()
    model = initialize_model()
    criterions = get_criterions()
    optimizer, scheduler = configure_optimizer(model)

    for epoch in range(1, epochs + 1):
        logging.info(f'Starting epoch {epoch}/{epochs}')
        train_loss = train(model, train_loader, criterions, optimizer, scheduler)
        val_loss = validate(model, val_loader, criterions)

        logging.info(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if epoch % save_model_interval == 0:
            save_model(epoch, model, optimizer, scheduler, val_loss)

    # Save final model
    save_model(epochs, model, optimizer, scheduler, val_loss)
    logging.info('Training complete!')

if __name__ == '__main__':
    main()