import torch
import torch.nn as nn
from torchvision import models

    
class PersonClassifierB3(nn.Module):

    def __init__(self, num_classes=9, pretrained=True):
        super().__init__()

        self.model = models.resnet50(weights="DEFAULT" if pretrained else None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)    
    
class GroupClassifierB3(nn.Module):

    def __init__(self,backbone,num_players=12,num_classes=8):
        super().__init__()

        self.backbone = backbone
        self.feature_dim = 2048

        self.num_players = num_players

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim,4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,num_classes)
        )



    def forward(self,x):

        # x:
        # batch, players, C,H,W

        B,P,C,H,W = x.shape

        x = x.view(B*P,C,H,W)

        features = self.backbone(x)
        
        features = torch.flatten(features,start_dim=1)
        
        features = features.view(B,P,-1)

        # reshape back to players dimension
        features = features.view(B, P, self.feature_dim)

        # max pooling over players
        features, _ = torch.max(features, dim=1)
        
        output = self.classifier(features)

        return output