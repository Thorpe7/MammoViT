import torch.nn as nn

class ChannelProjector(nn.Module):
    '''Project layer for dimensionality adjustment of input prior to entering custom ViT'''
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)

    def forward(self, x):
        return self.projection(x)
