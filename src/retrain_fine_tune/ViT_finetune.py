import torch
import torch.nn as nn
import os

def fine_tune_model(model, train_loader, val_loader, epochs=5, lr=1e-4, checkpoint_path=None):
    """
    Fine-tune the model on a new dataset. Optionally resume from a checkpoint and save progress.
    
    Args:
        model (nn.Module): The model to fine-tune.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        checkpoint_path (str or Path, optional): Path to save/load checkpoint.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    # Resume from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}")

        # Save checkpoint
        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
