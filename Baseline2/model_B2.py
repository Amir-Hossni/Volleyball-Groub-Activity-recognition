import torch
import torch.nn as nn
import torchvision.models as models


class B2Model(nn.Module):

    def __init__( self, num_classes=8,num_players=12, pretrained=True ):
        super().__init__()

        # CNN feature extractor
        resnet = models.resnet50(weights="DEFAULT" if pretrained else None)
        
        # remove classification layer
        self.cnn = nn.Sequential( *list(resnet.children())[:-1])

        self.feature_dim = 2048
        self.num_players = num_players

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),

            nn.Linear(self.feature_dim * self.num_players, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, num_classes)
        )


    def forward(self, x):

        """
        x:
        batch_size, 12, 3,224,224
        """

        batch_size = x.size(0)

        # Process every player
        # =========================

        x = x.view(batch_size * self.num_players,3,224,224)
        features = self.cnn(x)
        # ResNet output:
        # (B*12,2048,1,1)

        features = torch.flatten(features,start_dim=1)

        # back to players

        features = features.view(
            batch_size,
            self.num_players,
            self.feature_dim)
        
        # Concatenate players
        features = features.reshape(batch_size,-1)

        # Classification
        output = self.classifier(features)


        return output