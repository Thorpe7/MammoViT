import torchvision.models as models
import torch.nn as nn

class ResNet:
    def __init__(self):
        # Load ResNet50 model with IMAGENET1K_V2 weights
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the top layers (global pooling and dense layer)
        self.model = nn.Sequential(*list(self.model.children())[:-2])

    def get_model(self):
        # Return the modified model
        return self.model
