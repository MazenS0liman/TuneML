# ——————————————————————————————————————————————————————————————
# Imports
from typing import Dict, Optional, Tuple

import torch
from torch import nn

# ——————————————————————————————————————————————————————————————
# VGG Block
class VGGBlock(nn.Module):
    """
    A VGG block consisting of two convolutional layers followed by a max pooling layer.
    Each convolutional layer is followed by a ReLU activation function.
    """
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)      # Output shape: (out_channels, H, W)
        self.bn1 = nn.BatchNorm2d(out_channels)                                                   # Batch normalization after the first convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)     # Output shape: (out_channels, H, W)
        self.bn2 = nn.BatchNorm2d(out_channels)                                                   # Batch normalization after the second convolution
        self.relu = nn.ReLU()                                                            # Activation function applied after each convolution
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                                # Max pooling reduces spatial dimensions by a factor of 2 (output shape: (out_channels, H/2, W/2))

    def forward(self, x, fmap_dict=None, prefix=""):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.pool(x2)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}_conv1"] = x1
            fmap_dict[f"{prefix}_conv2"] = x2
            fmap_dict[f"{prefix}_pool"] = x3
        return x3

# ——————————————————————————————————————————————————————————————
# MiniVGG Model
class MiniVGG(nn.Module):
    """
    A simplified version of the VGG architecture for audio classification.
    """
    def __init__(self, num_classes=50, target_h=128, target_w=128):
        super(MiniVGG, self).__init__()
        # Define the first VGG block (input channels: 1 for grayscale spectrograms, output channels: 32)
        self.block1 = VGGBlock(in_channels=1, out_channels=32)
        # Define the second VGG block (input channels: 32, output channels: 64)
        self.block2 = VGGBlock(in_channels=32, out_channels=64)
        # Define the third VGG block (input channels: 64, output channels: 128)
        self.block3 = VGGBlock(in_channels=64, out_channels=128)
        # Define the fully connected layer
        self.fc1 = nn.Linear(128 * (target_h // 8) * (target_w // 8), num_classes)  # Calculate the input features based on the output of the last VGG block

        self.flatten = nn.Flatten()  # Flatten layer to convert 2D feature maps into a 1D vector for the fully connected layer
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization

    def forward(self, x, return_feature_maps=False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if not return_feature_maps:            
            fmap_dict = {} if return_feature_maps else None
            x = self.block1(x, fmap_dict, prefix="block1")
            x = self.block2(x, fmap_dict, prefix="block2")
            x = self.block3(x, fmap_dict, prefix="block3")
            x = self.flatten(x)         # Flatten the output of the convolutional blocks to a 1D vector
            x = self.dropout(x)         # Apply dropout for regularization
            out = self.fc1(x)           # Pass the flattened features through the fully connected layer to get class scores

            return out

        else:
            feature_maps = {}
            
            x = self.block1(x, feature_maps, prefix="block1")
            x = self.block2(x, feature_maps, prefix="block2")
            x = self.block3(x, feature_maps, prefix="block3")
            x = self.flatten(x)
            x = self.dropout(x)
            out = self.fc1(x)
            
            return out, feature_maps
            