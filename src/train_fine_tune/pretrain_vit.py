import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

from src.models.ViT import ViTModel
from src.train_fine_tune.metric_collection import initialize_metric_logs, log_epoch_metrics, save_confusion_matrix
from src.data_loading.torch_data_utils import TransformSubset


# -- Custom Dataset to ignore .ipynb_checkpoints --
class CleanImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name != '.ipynb_checkpoints']
        classes.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx


# -- Projects RGB image to 8-channel image expected by ViT --
class ProjectToEightChannels(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(3, 8, kernel_size=1)

    def forward(self, x):
        return self.project(x)


# -- Dataloader creation with stratified split --
def create_dataloaders(data_dir, batch_size=32, num_workers=2, val_split=0.2):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    base_dataset = CleanImageFolder(root=data_dir)
    targets = [label for _, label in base_dataset.samples]
    indices = range(len(base_dataset))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(indices, targets)) # type: ignore

    train_subset = TransformSubset(Subset(base_dataset, train_idx), train_transform)
    val_subset = TransformSubset(Subset(base_dataset, val_idx), val_transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# -- Main training loop --
def train_vit_model(data_dir, log_dir, epochs=10, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(data_dir, batch_size)
    model = ViTModel(num_classes=1000).to(device)
    projector = ProjectToEightChannels().to(device)
    optimizer = Adam(list(model.parameters()) + list(projector.parameters()), lr=learning_rate)
    criterion = CrossEntropyLoss()

    log_dir = Path(log_dir)
    initialize_metric_logs(log_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        projector.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = projector(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total

        # Validation
        model.eval()
        projector.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = projector(inputs)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss /= val_total

        log_epoch_metrics(log_dir, epoch, train_loss, val_loss, train_acc, val_acc)
        save_confusion_matrix(log_dir, all_labels, all_preds)

        print(f"[Epoch {epoch}/{epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
