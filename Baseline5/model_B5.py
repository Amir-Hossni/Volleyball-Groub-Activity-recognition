import torch
import torch.nn as nn
from torchvision import models




class PersonTemporalB5(nn.Module):

    def __init__(
            self,
            backbone,
            num_classes=9,
            lstm_hidden=512,
            lstm_layers=1,
            dropout=0.2
        ):
        
        super().__init__()

        # B3 Stage-A backbone
        self.backbone = backbone
        # Remove B3's classification head
        self.backbone.fc = nn.Identity()
        self.feature_dim = 2048
        
        #freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.num_players = num_classes
        self.lstm = nn.LSTM(
                    input_size=self.feature_dim,   # 2048
                    hidden_size=lstm_hidden, # 512
                    num_layers=lstm_layers,
                    batch_first=True,
                    dropout=dropout if lstm_layers > 1 else 0.0
                )
        self.classifier = nn.Linear(lstm_hidden, num_classes)


    def forward(self,x, return_features=False):

        # x:
        # (B, T, C, H, W)

        B,T,C,H,W = x.shape

        # merge time with batch
        x = x.view(B * T, C, H, W)
        
        # CNN is frozen
        with torch.no_grad():
            features = self.backbone(x)

        # feats:
        # (B*T,2048)

        # restore time dimension
        features = features.view(B, T, self.feature_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)

        # last layer hidden state
        final_feature = h_n[-1]
        
        if return_features:
            return final_feature
        
        # (B,512)
        output = self.classifier(final_feature)

        return output