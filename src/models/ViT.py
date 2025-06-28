import timm
import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
from src.retrain_fine_tune.ViT_finetune import fine_tune_model  # Import the new function

# Configure logging
log_dir = Path().cwd() / "logs" / "inference"
if not log_dir.exists():
    os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "inference_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ViTModel:
    def __init__(self, num_classes=1000):
        # Load the pre-trained model
        self.model = timm.create_model('deit_base_patch16_224', pretrained=True)
        # Modify the classifier for the specified number of classes
        self.model.head = nn.Linear(self.model.head.in_features, num_classes) # type: ignore

    def inference(self, inputs):
        """
        Perform inference with the model.
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).
        Returns:
            torch.Tensor: Model predictions.
        """
        self.model.eval()
        try:
            with torch.no_grad():
                outputs = self.model(inputs)
            return outputs
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            raise e

    def fine_tune(self, train_loader, val_loader, epochs=5, lr=1e-4, checkpoint_path=None):
        """
        Fine-tune the model on a new dataset, with optional checkpointing.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Total number of epochs to train.
            lr (float): Learning rate.
            checkpoint_path (str, optional): Path to load from or save checkpoint.
        """
        fine_tune_model(
            self.model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            checkpoint_path=checkpoint_path
        )