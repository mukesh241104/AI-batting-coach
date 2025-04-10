# i3d_model.py
import torch.nn as nn
import torch

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(Simple3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 16, 224, 224]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # [B, 32, 16, 112, 112]

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # [B, 64, 8, 56, 56]

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [B, 128, 1, 1, 1]
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)

def get_i3d_model(num_classes=5):
    return Simple3DCNN(num_classes)
