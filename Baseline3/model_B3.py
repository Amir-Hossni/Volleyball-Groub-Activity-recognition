
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


# def get_device(prefer_cuda=True):
#     """Single source of truth for the device. Use this everywhere -
#     model, trainer, evaluator - so CPU/GPU never disagree."""
#     if prefer_cuda and torch.cuda.is_available():
#         return torch.device("cuda")
#     return torch.device("cpu")
    
# class PersonClassifierB3(nn.Module):
#     """Baseline 3: person-level action classifier.

#     Input : cropped person bounding boxes  (B, 3, H, W)
#     Output: logits over the 9 individual action classes, or the pooled
#             2048-d backbone features when return_features=True.
#     """

#     def __init__(
#         self,
#         num_classes=9,
#         pretrained=True,
#         freeze_backbone=False,
#     ):
#         super().__init__()

#         # Pinned checkpoint - "DEFAULT" silently changes across torchvision
#         # releases, which breaks reproducibility between baselines.
#         weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

#         self.backbone = models.resnet50(weights=weights)

#         self.feature_dim = self.backbone.fc.in_features  # 2048
#         self.backbone.fc = nn.Identity()

#         self.classifier = nn.Linear(self.feature_dim, num_classes)

#         if freeze_backbone:
#             self.freeze_backbone()

#     def freeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#     def unfreeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = True

#     def forward(self, x, return_features=False):
#         features = self.backbone(x)

#         if return_features:
#             return features

#         return self.classifier(features)

#     @torch.no_grad()
#     def extract_features(self, x):
#         """Frozen feature extraction for B4+ (temporal / hierarchical stages)."""
#         self.eval()
#         return self.backbone(x)


# def build_b3(num_classes=9, pretrained=True, freeze_backbone=False, device=None):
#     """Build the model already placed on the right device."""
#     device = device if device is not None else get_device()

#     model = PersonClassifierB3(
#         num_classes=num_classes,
#         pretrained=pretrained,
#         freeze_backbone=freeze_backbone,
#     )

#     return model.to(device)


class PersonClassifierB3(nn.Module):

    def __init__(self, num_classes=9, pretrained=True):
        super().__init__()

        self.model = models.resnet50(weights="DEFAULT" if pretrained else None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        B, P, C, H, W = x.shape
        x = x.reshape(B * P, C, H, W)
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