import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from pathlib import Path
import os
import requests
import tarfile

from src.models.ViT import ViTModel
from src.train_fine_tune.metric_collection import initialize_metric_logs, log_epoch_metrics, save_confusion_matrix


# 🆕 Add ChannelProjector module
class ChannelProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)

    def forward(self, x):
        return self.projection(x)


def download_and_uncompress_imagenet(data_dir, archive_url, archive_name):
    """
    Download and uncompress the ImageNet archive if not already present.
    """
    archive_path = os.path.join(data_dir, archive_name)
    if not os.path.exists(archive_path):
        print(f"Downloading ImageNet archive from {archive_url}...")
        response = requests.get(archive_url, stream=True)
        with open(archive_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download completed.")

    if not os.path.exists(os.path.join(data_dir, 'train')) or not os.path.exists(os.path.join(data_dir, 'val')):
        print("Uncompressing ImageNet archive...")
        try:
            if archive_name.endswith('.tar.gz'):
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(path=data_dir)
            else:
                raise ValueError(f"Unsupported archive format: {archive_name}")
        except tarfile.ReadError as e:
            print(f"Error uncompressing archive: {e}")
            print("Please verify the downloaded file or download it manually.")
            raise
        print("Uncompression completed.")


def prepare_imagenet_dataloader(data_dir, batch_size):
    """
    Download ImageNet-1k and create dataloaders.
    """
    archive_url = "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz"
    archive_name = "ILSVRC2012_devkit_t12.tar.gz"
    download_and_uncompress_imagenet(data_dir, archive_url, archive_name)  # 🆕 Ensure archive is downloaded and uncompressed

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageNet(root=data_dir, split='train', download=False, transform=transform)
    val_dataset = datasets.ImageNet(root=data_dir, split='val', download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_vit_model(log_dir, save_path, epochs=10, batch_size=32, learning_rate=1e-4, data_dir='./data'):
    """
    Train the custom ViT model on ImageNet-1k and log metrics.
    Save the final trained state for later fine-tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🆕 Use the new function to prepare dataloaders
    train_loader, val_loader = prepare_imagenet_dataloader(data_dir, batch_size)

    projector = ChannelProjector().to(device)
    model = ViTModel(num_classes=1000).to(device)

    optimizer = Adam(list(projector.parameters()) + list(model.parameters()), lr=learning_rate)
    criterion = CrossEntropyLoss()

    log_dir = Path(log_dir)
    initialize_metric_logs(log_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        projector.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = projector(inputs)  # 🆕 project to 8 channels
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
        projector.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = projector(inputs)  # 🆕 apply projection during eval too
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

        print(f"Epoch {epoch}/{epochs} completed.")
        print(f"Training Metrics: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")
        print(f"Validation Metrics: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")

    # 🆕 Save the final trained state
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
