import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.optim as optim
import numpy as np

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_labels=2, dropout_rate=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Take the pooled output from the BERT model
        dropped_out = self.dropout(pooled_output)
        logits = self.classifier(dropped_out)
        return logits

def prepare_input(text, tokenizer, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoding['input_ids'], encoding['attention_mask']

def calculate_accuracy(preds, labels):
    _, predicted_labels = torch.max(preds, dim=1)
    correct_predictions = torch.sum(predicted_labels == labels)
    return correct_predictions.item() / len(labels)

def train_model(model, data_loader, optimizer, criterion, device):
    model = model.train()
    total_loss = 0
    total_accuracy = 0

    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        model.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        acc = calculate_accuracy(logits, labels)
        total_accuracy += acc

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    return avg_loss, avg_accuracy

def evaluate_model(model, data_loader, criterion, device):
    model = model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            acc = calculate_accuracy(logits, labels)
            total_accuracy += acc

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    return avg_loss, avg_accuracy

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids, attention_mask = prepare_input(text, self.tokenizer, self.max_length)
        return {
            "input_ids": input_ids.flatten(),
            "attention_mask": attention_mask.flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(texts, labels, tokenizer, max_length, batch_size):
    dataset = CustomDataset(texts, labels, tokenizer, max_length)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BERTClassifier(num_labels=3)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    texts = [
        "This is a positive review.",
        "This is a negative review.",
        "This is a neutral review."
    ]
    
    labels = [0, 1, 2]  # 0 = positive, 1 = negative, 2 = neutral
    
    data_loader = create_data_loader(texts, labels, tokenizer, max_length=128, batch_size=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(2):  # Train for 2 epochs
        train_loss, train_acc = train_model(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{2}")
        print(f"Train loss: {train_loss}, Train accuracy: {train_acc}")

    # Create validation data
    val_texts = [
        "This is another positive review.",
        "This is another negative review.",
        "This is another neutral review."
    ]
    val_labels = [0, 1, 2]  # Labels corresponding to the validation data

    val_data_loader = create_data_loader(val_texts, val_labels, tokenizer, max_length=128, batch_size=2)

    for epoch in range(2):  # Train and evaluate for 2 epochs
        train_loss, train_acc = train_model(model, data_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_data_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{2}")
        print(f"Train loss: {train_loss}, Train accuracy: {train_acc}")
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

    # Save the trained model
    torch.save(model.state_dict(), "bert_classifier_model.pth")
    print("Model saved to bert_classifier_model.pth")

    # Inference function for making predictions on new text
    def predict(model, tokenizer, text, max_length=128):
        model = model.eval()
        input_ids, attention_mask = prepare_input(text, tokenizer, max_length)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        return np.argmax(probabilities), probabilities

    # Make predictions on some test sentences
    test_texts = [
        "This is a test sentence.",
        "I don't like this at all.",
        "What a wonderful day!"
    ]

    for test_text in test_texts:
        predicted_class, probabilities = predict(model, tokenizer, test_text)
        print(f"Text: {test_text}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Probabilities: {probabilities}")

    # Function to load the saved model
    def load_model(model_path, num_labels=3):
        model = BERTClassifier(num_labels=num_labels)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        return model

    # Load the model and test inference
    loaded_model = load_model("bert_classifier_model.pth")
    for test_text in test_texts:
        predicted_class, probabilities = predict(loaded_model, tokenizer, test_text)
        print(f"After loading, Text: {test_text}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Probabilities: {probabilities}")

    # Hyperparameter tuning (simplified version)
    def hyperparameter_tuning(num_epochs, learning_rates, dropout_rates):
        best_acc = 0
        best_model = None
        best_params = {}

        for lr in learning_rates:
            for dr in dropout_rates:
                print(f"Tuning with learning rate: {lr}, dropout rate: {dr}")
                model = BERTClassifier(dropout_rate=dr)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                model = model.to(device)
                for epoch in range(num_epochs):
                    train_loss, train_acc = train_model(model, data_loader, optimizer, criterion, device)
                    val_loss, val_acc = evaluate_model(model, val_data_loader, criterion, device)

                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model = model.state_dict()
                        best_params = {'learning_rate': lr, 'dropout_rate': dr}

        print(f"Best Validation Accuracy: {best_acc}")
        print(f"Best Parameters: {best_params}")
        torch.save(best_model, "best_bert_model.pth")

    # Running hyperparameter tuning with a list of learning rates and dropout rates
    learning_rates = [2e-5, 3e-5, 5e-5]
    dropout_rates = [0.1, 0.2, 0.3]

    hyperparameter_tuning(2, learning_rates, dropout_rates)

    # Function to load the best model after hyperparameter tuning
    def load_best_model(model_path, num_labels=3):
        model = BERTClassifier(num_labels=num_labels)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        return model

    # Load the best model after hyperparameter tuning
    best_model = load_best_model("best_bert_model.pth")

    # Run predictions on the test data with the best model
    for test_text in test_texts:
        predicted_class, probabilities = predict(best_model, tokenizer, test_text)
        print(f"Best Model Prediction - Text: {test_text}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Probabilities: {probabilities}")

    # Defining learning rate scheduler for dynamic learning rate adjustments during training
    def create_scheduler(optimizer, num_warmup_steps, num_training_steps):
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) / num_warmup_steps, 1) * (1 - (step / num_training_steps))
        )
        return scheduler

    # Using learning rate scheduler
    num_training_steps = len(data_loader) * 2  # Number of epochs
    num_warmup_steps = 0.1 * num_training_steps

    scheduler = create_scheduler(optimizer, num_warmup_steps, num_training_steps)

    for epoch in range(2):
        model.train()
        for step, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()  

    from torch.cuda.amp import autocast, GradScaler

    # Implementing gradient clipping to avoid exploding gradients
    def train_model_with_clipping(model, data_loader, optimizer, criterion, device, max_grad_norm=1.0):
        model.train()
        total_loss = 0
        total_accuracy = 0

        for step, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calculate accuracy
            acc = calculate_accuracy(logits, labels)
            total_accuracy += acc

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_accuracy / len(data_loader)
        return avg_loss, avg_accuracy

    # Usage of gradient clipping during training
    for epoch in range(2):
        train_loss, train_acc = train_model_with_clipping(model, data_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_data_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{2} with gradient clipping")
        print(f"Train loss: {train_loss}, Train accuracy: {train_acc}")
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

    # Mixed precision training for faster training and reduced memory usage
    scaler = GradScaler()

    def train_model_with_mixed_precision(model, data_loader, optimizer, criterion, device, scaler):
        model.train()
        total_loss = 0
        total_accuracy = 0

        for step, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            # Scales the loss to prevent underflow
            scaler.scale(loss).backward()

            # Gradient clipping for safety
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            acc = calculate_accuracy(logits, labels)
            total_accuracy += acc

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_accuracy / len(data_loader)
        return avg_loss, avg_accuracy

    # Using mixed precision training
    for epoch in range(2):
        train_loss, train_acc = train_model_with_mixed_precision(model, data_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate_model(model, val_data_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{2} with mixed precision")
        print(f"Train loss: {train_loss}, Train accuracy: {train_acc}")
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

    # Adding support for custom tokenization pipelines, such as adding special tokens or handling rare characters

    def custom_tokenization(texts, tokenizer, special_tokens=None, max_length=128):
        """Applies custom tokenization, including special tokens."""
        encoded_inputs = []
        for text in texts:
            if special_tokens:
                text = special_tokens[0] + text + special_tokens[1]  # Insert special tokens at the start and end

            input_ids, attention_mask = prepare_input(text, tokenizer, max_length)
            encoded_inputs.append({
                "input_ids": input_ids.flatten(),
                "attention_mask": attention_mask.flatten()
            })

        return encoded_inputs

    # Custom tokenization with special tokens
    special_tokens = ["[SPECIAL_START]", "[SPECIAL_END]"]
    test_texts_with_special_tokens = custom_tokenization(test_texts, tokenizer, special_tokens=special_tokens)

    for encoded in test_texts_with_special_tokens:
        print(f"Input IDs with special tokens: {encoded['input_ids']}")
        print(f"Attention Mask: {encoded['attention_mask']}")

    # Custom training loop with dynamic learning rate, gradient clipping, and mixed precision training
    def custom_training_loop(model, data_loader, val_data_loader, optimizer, criterion, device, scheduler=None, max_grad_norm=1.0, num_epochs=2):
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            total_accuracy = 0

            for step, batch in enumerate(data_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                acc = calculate_accuracy(logits, labels)
                total_accuracy += acc

                # Update scheduler if provided
                if scheduler:
                    scheduler.step()

            avg_loss = total_loss / len(data_loader)
            avg_accuracy = total_accuracy / len(data_loader)
            val_loss, val_acc = evaluate_model(model, val_data_loader, criterion, device)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {avg_loss}, Train Accuracy: {avg_accuracy}")
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    # Usage of the custom training loop
    custom_training_loop(
        model, data_loader, val_data_loader, optimizer, criterion, device,
        scheduler=scheduler, max_grad_norm=1.0, num_epochs=2
    )

    # Additional utility functions to log performance metrics, save model checkpoints during training
    def save_checkpoint(model, optimizer, epoch, filepath="checkpoint.pth"):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at epoch {epoch} to {filepath}")

    def load_checkpoint(filepath, model, optimizer):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {filepath} at epoch {epoch}")
        return epoch

    # Saving and loading checkpoints
    save_checkpoint(model, optimizer, epoch=2, filepath="bert_checkpoint.pth")
    loaded_epoch = load_checkpoint("bert_checkpoint.pth", model, optimizer)