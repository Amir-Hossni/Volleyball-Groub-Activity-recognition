import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
from torchvision import models


class SceneClassifierB1(nn.Module):

    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()

        self.model = models.resnet50(weights="DEFAULT" if pretrained else None)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)