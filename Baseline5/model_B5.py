import torch
import torch.nn as nn


class PersonTemporalB5(nn.Module):
    """Stage-A: per-player temporal action classifier.

    Frozen CNN backbone (feature extractor) -> LSTM over time -> linear head.
    """

    def __init__(
        self,
        backbone,
        num_classes=9,
        lstm_hidden=512,
        lstm_layers=1,
        dropout=0.2,
    ):
        super().__init__()

        # Stage-A backbone (frozen feature extractor, e.g. ResNet-50)
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.feature_dim = 2048

        # Freeze backbone weights...
        for param in self.backbone.parameters():
            param.requires_grad = False
        # ...and make sure it starts in eval mode so BatchNorm running
        # stats don't drift, and dropout (if any) is disabled.
        self.backbone.eval()

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,   # 2048
            hidden_size=lstm_hidden,       # 512
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen backbone in eval mode no matter what mode
        # the parent model is switched to.
        self.backbone.eval()
        return self

    def forward(self, x, return_features=False):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # merge time with batch
        x = x.view(B * T, C, H, W)

        # CNN is frozen
        with torch.no_grad():
            features = self.backbone(x)
        # features: (B*T, 2048)

        # restore time dimension
        features = features.view(B, T, self.feature_dim)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        # last layer's hidden state
        final_feature = h_n[-1]  # (B, 512)

        if return_features:
            return final_feature

        output = self.classifier(final_feature)  # (B, num_classes)
        return output


class GroupTemporalClassifierB5(nn.Module):
    """Stage-B: team-level classifier via concatenation of per-player features.

    Wraps a frozen Stage-A PersonTemporalB5, extracts per-player temporal
    features, concatenates them across players, and classifies the group
    activity.
    """

    def __init__(
        self,
        person_model,
        num_classes=8,
        num_players=12,
        hidden_dim=2048,
        dropout=0.4,
    ):
        super().__init__()

        # Stage-A temporal person model (frozen)
        self.person_model = person_model
        self.num_players = num_players
        self.player_feature_dim = 512

        for param in self.person_model.parameters():
            param.requires_grad = False
        self.person_model.eval()

        # Concatenation: MAX + MEAN
        # input_dim = (2 * self.player_feature_dim)

        # Full Concatenation:
        # 12 players × 512 features = 6144
        
        input_dim = self.num_players * self.player_feature_dim

        
        # Stage-B classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen Stage-A (and, transitively, its backbone) in eval mode.
        self.person_model.eval()
        return self

    def forward(self, x, player_mask):
        # x: (B, P, T, C, H, W)
        B, P, T, C, H, W = x.shape

        # merge batch + players
        x = x.reshape(B * P, T, C, H, W)
        
        
        # Stage-A inference (frozen)
        with torch.no_grad():
            player_features = self.person_model( x, return_features=True) # (B*P, 512)
            

        # restore player dimension
        player_features = player_features.reshape(B, P, self.player_feature_dim) # (B, P, 512)
        
        # (B,P) -> (B,P,1)
      
        # Apply player mask
        mask_3d = player_mask.unsqueeze(-1) # Convert to: (B, P, 1)

        # Zero-out padded / invalid players
        player_features = player_features.masked_fill(~mask_3d, 0.0)


        # Full Concatenation
        # (B, 12, 512)
        #       ↓
        # (B, 6144)
        team_features = player_features.reshape(
            B,
            self.num_players * self.player_feature_dim
        )

        # Stage-B classifier
        output = self.classifier(team_features)  # (B, 8)


        return output


