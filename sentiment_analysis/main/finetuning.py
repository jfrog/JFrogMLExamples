# Tokenize the data
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AdamW


# Define PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, idx):
            item = self.examples[idx]
            return {
                'input_ids': torch.tensor(item['input_ids']),
                'attention_mask': torch.tensor(item['attention_mask']),
                'label': torch.tensor(item['label']),
            }

    def __len__(self):
        return len(self.examples)


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)


def generate_dataset(tokenizer, dataset) -> tuple[CustomDataset, CustomDataset]:
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    # Train-validation split
    train_dataset, eval_dataset = (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
    )

    train_dataset = CustomDataset(train_dataset)
    eval_dataset = CustomDataset(eval_dataset)
    return train_dataset, eval_dataset

def eval_model(model, device, eval_loader):
    print("Running model evaluation")
    # Evaluation
    model.eval()
    eval_loss = 0
    total_eval_batches = len(eval_loader)
    for batch_idx, batch in enumerate(eval_loader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item()
        
        # Log every 10% of evaluation batches
        if batch_idx % (total_eval_batches // 10) == 0:
            print(f"Evaluation Batch {batch_idx}/{total_eval_batches}, Eval Loss: {eval_loss / (batch_idx + 1):.4f}")
    avg_eval_loss = eval_loss / total_eval_batches
    return avg_eval_loss

def train_model(
    model, device, lr, num_epochs, train_loader, eval_loader, early_stopping, logger
):
    # Early stopping configuration
    patience = 3
    best_eval_loss = float("inf")
    epochs_no_improve = 0
    log_interval = len(train_loader) // 10
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    # Fine-tuning loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
                    # Log every 10% of batches
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {train_loss / (batch_idx + 1):.4f}")
            # Log every 100 batches
            if batch_idx % 100 == 0 and batch_idx != 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {train_loss / (batch_idx + 1):.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)

        avg_eval_loss = eval_model(model, device, eval_loader)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}"
        )

        # Early stopping
        if early_stopping and avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
    return model
