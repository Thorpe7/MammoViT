import timm
import torch
import torch.nn as nn
import logging
import os
from pathlib import Path# Import the new function

# Configure logging
log_dir = Path().cwd() / "logs" / "inference"
if not log_dir.exists():
    os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "inference_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTModel:
    def __init__(self, num_classes=1000):
        # Load the pre-trained model
        self.model = timm.create_model('deit_base_patch16_224', pretrained=True)
        # Modify the classifier for the specified number of classes
        self.model.head = nn.Linear(self.model.head.in_features, num_classes) # type: ignore
        self.model.to(device)

    def inference(self, inputs):
        """
        Perform inference with the model.
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).
        Returns:
            torch.Tensor: Model predictions.
        """
        self.model.eval()
        inputs = inputs.to(device)
        try:
            with torch.no_grad():
                outputs = self.model(inputs)
            return outputs
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            raise e