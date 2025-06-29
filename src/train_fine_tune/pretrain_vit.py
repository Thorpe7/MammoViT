import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from pathlib import Path
from src.models.ViT import ViTModel
from src.train_fine_tune.metric_collection import initialize_metric_logs, log_epoch_metrics, save_confusion_matrix

def create_dataloaders(data_dir, batch_size=32):
    """
    Create dataloaders for ImageNet-1k training and validation datasets.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=data_dir / "train", transform=transform)
    val_dataset = datasets.ImageFolder(root=data_dir / "val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_vit_model(data_dir, log_dir, epochs=10, batch_size=32, learning_rate=1e-4):
    """
    Train the custom ViT model on ImageNet-1k and log metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders(data_dir, batch_size)
    model = ViTModel(num_classes=1000).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    log_dir = Path(log_dir)
    initialize_metric_logs(log_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss /= val_total

        log_epoch_metrics(log_dir, epoch, train_loss, val_loss, train_acc, val_acc)
        save_confusion_matrix(log_dir, all_labels, all_preds)

        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
