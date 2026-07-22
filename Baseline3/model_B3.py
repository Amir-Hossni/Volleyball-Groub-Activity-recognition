import torch
import torch.nn as nn
from torchvision import models


class PersonClassifierB3(nn.Module):

    def __init__(self,num_classes=9,pretrained=True):
        super().__init__()

        resnet = models.resnet50(weights="DEFAULT" if pretrained else None)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_dim = 2048

        self.classifier = nn.Linear(self.feature_dim,num_classes)


    def forward(self,x):

        features = self.backbone(x)

        features = features.view(features.size(0),-1)

        logits = self.classifier(features)


        return logits
    
    
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