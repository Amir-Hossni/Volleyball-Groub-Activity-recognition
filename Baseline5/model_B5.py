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
        hidden_dim=512,
        dropout=0.3,
    ):
        super().__init__()

        # Stage-A temporal person model (frozen)
        self.person_model = person_model
        self.num_players = num_players
        self.player_feature_dim = person_model.lstm_hidden

        for param in self.person_model.parameters():
            param.requires_grad = False
        self.person_model.eval()

        # Concatenation: MAX + MEAN
        input_dim = (2 * self.player_feature_dim)

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

    def forward(self, x):
        # x: (B, P, T, C, H, W)
        B, P, T, C, H, W = x.shape

        # merge batch + players
        x = x.reshape(B * P, T, C, H, W)

        # Stage-A inference (frozen)
        with torch.no_grad():
            player_features = self.person_model( x, return_features=True) # (B*P, 512)
            

        # restore player dimension
        player_features = player_features.reshape(B, P, self.player_feature_dim) # (B, P, 512)
        

        # MAX pooling across players
        pooled_max = player_features.max(dim=1)[0] # (B, 512)
        

        # MEAN pooling across players
        pooled_mean = player_features.mean(dim=1) # (B, 512)
        

        # concatenate MAX + MEAN
        team_features = torch.cat([pooled_max, pooled_mean], dim=-1) # (B, 1024)
        

        output = self.classifier(team_features) # (B, 8)
        

        return output


class GroupTemporalClassifierB5V2(nn.Module):
    """Stage-B (v2): team-level classifier built directly from a frozen
    backbone + LSTM (rather than wrapping a PersonTemporalB5 instance),
    with max-pooling aggregation across players.
    """

    def __init__(
        self,
        backbone,
        lstm,
        num_classes=8,
        hidden_dim=4096,
        dropout=0.2,
    ):
        super().__init__()

        # Stage-A components (frozen)
        self.backbone = backbone
        self.lstm = lstm
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False

        # Stage-B classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen Stage-A in evaluation mode
        self.backbone.eval()
        self.lstm.eval()
        return self

    def forward(self, x):
        # x: (B, P, T, C, H, W)
        B, P, T, C, H, W = x.shape

        # merge batch, players, and time all at once
        x = x.reshape(B * P * T, C, H, W)

        # Stage-1 CNN
        with torch.no_grad():
            features = self.backbone(x)
        # (B*P*T, 2048)
        features = features.reshape(B * P, T, 2048)

        # Stage-1 LSTM
        with torch.no_grad():
            _, (h_n, _) = self.lstm(features)
        player_features = h_n[-1]  # (B*P, 512)

        # restore player dimension
        player_features = player_features.reshape(B, P, 512)

        # max pooling over players
        team_features, _ = torch.max(player_features, dim=1)  # (B, 512)

        output = self.classifier(team_features)  # (B, num_classes)
        return output

    
    
