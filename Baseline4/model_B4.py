import torch
import torch.nn as nn
from torchvision import models


class TemporalImageClassifierB4(nn.Module):

    def __init__(
        self,
        backbone,
        num_classes=8,
        lstm_hidden=512,
        lstm_layers=1,
        dropout=0.2
    ):
        super().__init__()

      
        # CNN Backbone
        self.backbone = backbone
        self.feature_dim = 2048   
        # remove classifier
        self.backbone.fc = nn.Identity()
        
        # Temporal model
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,   # 2048
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

    
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # x:
        # (B,T,C,H,W)

        B, T, C, H, W = x.shape

        # merge time with batch
        x = x.view(B * T, C, H, W)
     
        # CNN features
        feats = self.backbone(x)

        # feats:
        # (B*T,2048)

        # restore time dimension
        feats = feats.view(B, T, self.feature_dim)

        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(feats)

        # last layer hidden state
        final_feature = h_n[-1]

        # (B,512)

        output = self.classifier(final_feature)

        return output