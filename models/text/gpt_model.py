import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# GPT Model Configuration Class
class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layers, n_heads, n_embd, dropout):
        """
        Initializes the GPT configuration with key parameters like:
        - vocab_size: Size of the vocabulary (number of tokens).
        - block_size: Maximum sequence length the model can handle.
        - n_layers: Number of transformer blocks (layers).
        - n_heads: Number of attention heads in the multi-head attention mechanism.
        - n_embd: Size of each embedding vector.
        - dropout: Dropout probability to prevent overfitting.
        """
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = dropout

# Main GPT Model Class
class GPT(nn.Module):
    def __init__(self, config):
        """
        Initializes the GPT model. The architecture includes:
        - Embedding layers for both tokens and positions.
        - Multiple transformer blocks.
        - Final layer normalization and a projection to vocabulary size.
        """
        super().__init__()
        self.config = config
        
        # Token and Position Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        
        # Final Layer Normalization and Linear Projection
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

        # Initialize Weights for all layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes the weights of the model. Linear and Embedding layers
        are initialized with a normal distribution (mean=0, std=0.02).
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        The forward pass through the GPT model. It computes:
        - Token embeddings based on input indices.
        - Adds position embeddings.
        - Passes through multiple transformer blocks.
        - Applies final layer normalization and projects to vocabulary size.

        If targets are provided, cross-entropy loss is computed for training.
        """
        b, t = idx.size()  # batch size and sequence length
        assert t <= self.config.block_size, "Sequence length exceeds block size"

        # Token and Position Embeddings
        token_embeddings = self.tok_emb(idx)  # (b, t, n_embd)
        position_embeddings = self.pos_emb[:, :t, :]  # (1, t, n_embd)
        x = self.drop(token_embeddings + position_embeddings)  # Apply dropout

        # Pass through each transformer block
        for block in self.blocks:
            x = block(x)

        # Final Layer Norm and Linear Projection
        x = self.ln_f(x)
        logits = self.head(x)  # (b, t, vocab_size)

        # If targets are provided, compute cross-entropy loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return logits

# Block Class for Transformer
class Block(nn.Module):
    def __init__(self, config):
        """
        Defines a transformer block, consisting of:
        - Layer Normalization for inputs.
        - Multi-Head Attention mechanism.
        - Layer Normalization and MLP (Feed-forward) after attention.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        """
        Forward pass through the block:
        - Layer Normalization.
        - Multi-Head Attention.
        - Another Layer Normalization.
        - MLP (Feed-forward layer).
        """
        # Self-attention with residual connection
        x = x + self.attn(self.ln1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        
        return x

# Multi-Head Attention Mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        """
        Multi-head attention implementation:
        - n_heads: Number of attention heads.
        - head_dim: Dimensionality of each attention head.
        - Linear projections for query, key, and value vectors.
        - Dropout for attention and residual connections.
        """
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.scale = self.head_dim ** -0.5

        # Linear layers for Query, Key, Value projections
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # Dropout layers for attention and residual connections
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        """
        The forward pass through multi-head attention:
        - Compute Query, Key, and Value vectors.
        - Scale the dot-product of query and key vectors.
        - Apply softmax to obtain attention scores.
        - Use attention scores to compute weighted sum of value vectors.
        """
        b, t, c = x.size()  # Batch size, sequence length, embedding size
        
        # Project inputs to Query, Key, Value matrices and split heads
        q = self.query(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # Apply softmax
        attn = self.attn_drop(attn)  # Apply attention dropout

        # Weighted sum of value vectors
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # Recombine heads
        
        # Apply residual dropout and projection
        y = self.resid_drop(self.proj(y))
        
        return y

    def forward(self, x):
        """
        The forward pass through multi-head attention:
        - Compute Query, Key, and Value vectors.
        - Scale the dot-product of query and key vectors.
        - Apply softmax to obtain attention scores.
        - Use attention scores to compute weighted sum of value vectors.
        """
        b, t, c = x.size()  # Batch size, sequence length, embedding size
        
        # Project inputs to Query, Key, Value matrices and split heads
        q = self.query(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # Apply softmax
        attn = self.attn_drop(attn)  # Apply attention dropout

        # Weighted sum of value vectors
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # Recombine heads
        
        # Apply residual dropout and projection
        y = self.resid_drop(self.proj(y))
        
        return y

# Model for Language Generation Task
class GPTForLanguageGeneration(nn.Module):
    def __init__(self, config):
        """
        This class builds on the GPT model to fine-tune it for a language generation task.
        Implements additional components like a language model head that can generate
        text conditioned on previous inputs.
        """
        super().__init__()
        self.gpt = GPT(config)

    def forward(self, input_ids, targets=None):
        """
        Forward pass to generate logits. If targets are provided, also compute loss.
        - input_ids: Tokenized input sequences.
        - targets: The expected output, used to calculate the loss.
        """
        logits, loss = self.gpt(input_ids, targets)
        if targets is not None:
            return logits, loss
        return logits

# Training loop for the GPT model
def train_gpt_model(model, dataloader, optimizer, scheduler, device):
    """
    The training loop function for the GPT model:
    - model: The GPT model to be trained.
    - dataloader: A PyTorch DataLoader that provides input batches for training.
    - optimizer: The optimizer to update the model's weights.
    - scheduler: A learning rate scheduler for adjusting the learning rate dynamically.
    - device: The device (CPU or GPU) where the model and data are loaded.
    """
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids, targets = batch['input_ids'].to(device), batch['targets'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        logits, loss = model(input_ids, targets)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Function to evaluate GPT model
def evaluate_gpt_model(model, dataloader, device):
    """
    Function to evaluate the GPT model on validation or test data:
    - model: The trained GPT model.
    - dataloader: A PyTorch DataLoader for validation or testing data.
    - device: The device (CPU or GPU) where the model and data are loaded.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, targets = batch['input_ids'].to(device), batch['targets'].to(device)

            # Forward pass through the model
            logits, loss = model(input_ids, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Function for Text Generation with GPT Model
def generate_text(model, tokenizer, prompt, max_length, temperature=1.0, top_k=None, top_p=None):
    """
    Generate text based on a given prompt using the trained GPT model.
    - model: The fine-tuned GPT model.
    - tokenizer: Tokenizer to encode/decode input/output.
    - prompt: The initial text prompt to begin generation.
    - max_length: Maximum length of the generated sequence.
    - temperature: Sampling temperature to control randomness.
    - top_k: Number of top k tokens to sample from.
    - top_p: Top cumulative probability for sampling.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            logits = outputs[:, -1, :] / temperature

            if top_k is not None:
                logits = top_k_logits(logits, top_k)

            if top_p is not None:
                logits = top_p_logits(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

def top_k_logits(logits, k):
    """
    Filters the logits to retain only the top-k highest values.
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def top_p_logits(logits, p):
    """
    Filters the logits to retain only tokens whose cumulative probability is below the top-p threshold.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')

    return logits

# Optimizer and Scheduler Setup
def setup_optimizer_scheduler(model, learning_rate, weight_decay, total_steps):
    """
    Sets up the AdamW optimizer and learning rate scheduler for training:
    - model: The GPT model.
    - learning_rate: The initial learning rate.
    - weight_decay: The weight decay rate for regularization.
    - total_steps: Total number of steps for the scheduler.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    return optimizer, scheduler

# Tokenizer Setup
def setup_tokenizer(vocab_file, merges_file):
    """
    Loads a tokenizer from pre-trained vocab and merges files.
    - vocab_file: The vocabulary file.
    - merges_file: The merges file for byte-pair encoding (BPE).
    """
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merges_file)

    return tokenizer

# Function to Save Model Checkpoints
def save_model_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Saves the model checkpoint, including the model state, optimizer state, 
    scheduler state, and the current epoch.
    - model: The GPT model to be saved.
    - optimizer: The optimizer used during training.
    - scheduler: The learning rate scheduler.
    - epoch: Current epoch number.
    - path: The file path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)

# Function to Load Model Checkpoint
def load_model_checkpoint(model, optimizer, scheduler, path):
    """
    Loads the model checkpoint from a file and restores the model, optimizer, 
    and scheduler states.
    - model: The GPT model to load the state into.
    - optimizer: The optimizer whose state will be restored.
    - scheduler: The scheduler whose state will be restored.
    - path: The file path to load the checkpoint from.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

# Gradient Clipping for Stability
def apply_gradient_clipping(model, max_norm):
    """
    Applies gradient clipping to the model's parameters to prevent exploding 
    gradients during backpropagation.
    - model: The GPT model.
    - max_norm: The maximum allowable norm for the gradients.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Function to Calculate Perplexity
def calculate_perplexity(loss):
    """
    Calculates perplexity based on the model's loss.
    - loss: The cross-entropy loss from the model.
    """
    return torch.exp(loss)

# Model Configuration Loading
def load_config(config_path):
    """
    Loads the GPT model configuration from a given YAML or JSON file.
    - config_path: The file path to the configuration file.
    """
    import yaml
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return GPTConfig(**config_dict)

# Scheduler Step for Warmup
def warmup_scheduler_step(scheduler, step, warmup_steps):
    """
    Perform a step of the scheduler during the warmup phase of training.
    - scheduler: The learning rate scheduler.
    - step: The current training step.
    - warmup_steps: The total number of warmup steps.
    """
    if step < warmup_steps:
        lr_scale = min(1.0, float(step) / float(warmup_steps))
        for param_group in scheduler.optimizer.param_groups:
            param_group['lr'] = lr_scale * scheduler.base_lrs[0]
    else:
        scheduler.step()

from torch.cuda.amp import GradScaler, autocast

# Mixed Precision Training Setup
def setup_mixed_precision_training(model):
    """
    Sets up mixed precision training for the model, allowing more efficient 
    memory usage and faster computation by utilizing lower precision floats.
    - model: The GPT model.
    """
    

    scaler = GradScaler()
    return model, scaler

# Mixed Precision Training Loop
def mixed_precision_train_step(model, batch, optimizer, scaler, device):
    """
    Performs a single training step using mixed precision to reduce memory usage 
    and accelerate computations.
    - model: The GPT model.
    - batch: The input batch containing tokenized sequences.
    - optimizer: The optimizer used for gradient descent.
    - scaler: The gradient scaler for mixed precision.
    - device: The device (CPU or GPU) to run the computation.
    """
    input_ids, targets = batch['input_ids'].to(device), batch['targets'].to(device)
    
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with autocast():
        logits, loss = model(input_ids, targets)

    # Backward pass with mixed precision
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

# Function to Log Training Progress
def log_training_progress(epoch, step, loss, perplexity, total_steps):
    """
    Logs the training progress to monitor metrics such as loss and perplexity.
    - epoch: The current training epoch.
    - step: The current step within the epoch.
    - loss: The current loss value.
    - perplexity: The current perplexity value.
    - total_steps: The total number of steps in the epoch.
    """
    print(f"Epoch [{epoch}], Step [{step}/{total_steps}], Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")

# Early Stopping Mechanism
class EarlyStopping:
    def __init__(self, patience, min_delta=0):
        """
        Initializes the early stopping mechanism to halt training if validation 
        loss doesn't improve within a given patience period.
        - patience: The number of epochs to wait for an improvement.
        - min_delta: Minimum improvement threshold to count as progress.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Checks whether validation loss has improved and decides if early stopping 
        should be triggered.
        - val_loss: The current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Learning Rate Decay Function
def learning_rate_decay(optimizer, decay_factor):
    """
    Reduces the learning rate by a specified decay factor.
    - optimizer: The optimizer whose learning rate needs adjustment.
    - decay_factor: Factor by which to reduce the learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

# Cosine Annealing Scheduler
def cosine_annealing_scheduler(optimizer, T_max):
    """
    Applies cosine annealing scheduling to adjust the learning rate during 
    training, gradually reducing the rate after each cycle.
    - optimizer: The optimizer to adjust the learning rate for.
    - T_max: The number of steps over which to reduce the learning rate.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return scheduler

# Model Freezing for Transfer Learning
def freeze_model_parameters(model):
    """
    Freezes all parameters of the model to prevent them from being updated during 
    backpropagation, useful for transfer learning scenarios.
    - model: The GPT model whose parameters will be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False