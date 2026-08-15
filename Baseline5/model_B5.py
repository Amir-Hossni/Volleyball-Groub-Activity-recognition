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
        player_feature_dim=512,
        hidden_dim=4096,
        dropout=0.2,
    ):
        super().__init__()

        # Stage-A temporal person model (frozen)
        self.person_model = person_model
        self.num_players = num_players
        self.player_feature_dim = player_feature_dim

        for param in self.person_model.parameters():
            param.requires_grad = False
        self.person_model.eval()

        # Concatenation: (B, num_players, player_feature_dim) -> (B, input_dim)
        input_dim = num_players * player_feature_dim

        # Stage-B classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes),
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
            player_features = self.person_model(x, return_features=True)
            # (B*P, player_feature_dim)

        # restore player dimension
        player_features = player_features.reshape(B, P, self.player_feature_dim)
        # (B, num_players, player_feature_dim)

        # concatenate all player representations
        team_features = player_features.reshape(B, P * self.player_feature_dim)
        # (B, num_players * player_feature_dim)

        output = self.classifier(team_features)  # (B, num_classes)
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

    
    
    
# Best model saved (Val Accuracy = 76.53%)
# Confusion matrix saved to /kaggle/working/confusion_matrix_val.png
# Epoch 2/50, Train Loss: 0.4809 Train Accuracy: 83.68% Validation Loss: 0.7564 Validation Accuracy: 75.80% Validation F1: 0.4958
# =========================
# Epoch 3/50, Train Loss: 0.4276 Train Accuracy: 85.36% Validation Loss: 0.7776 Validation Accuracy: 75.15% Validation F1: 0.4936
# =========================
# Epoch 4/50, Train Loss: 0.3909 Train Accuracy: 86.36% Validation Loss: 0.8045 Validation Accuracy: 74.49% Validation F1: 0.4962
# =========================
# Epoch 5/50, Train Loss: 0.3594 Train Accuracy: 87.29% Validation Loss: 0.8310 Validation Accuracy: 73.93% Validation F1: 0.4990
# =========================
# Epoch 6/50, Train Loss: 0.3318 Train Accuracy: 88.22% Validation Loss: 0.8592 Validation Accuracy: 73.76% Validation F1: 0.4974
# =========================
# Epoch 7/50, Train Loss: 0.3070 Train Accuracy: 89.06% Validation Loss: 0.8823 Validation Accuracy: 73.52% Validation F1: 0.5019
# =========================
# Epoch 8/50, Train Loss: 0.2788 Train Accuracy: 90.12% Validation Loss: 0.9247 Validation Accuracy: 73.33% Validation F1: 0.4919
# =========================
# Epoch 9/50, Train Loss: 0.2523 Train Accuracy: 91.10% Validation Loss: 0.9600 Validation Accuracy: 73.02% Validation F1: 0.4957
# =========================
# Epoch 10/50, Train Loss: 0.2249 Train Accuracy: 92.21% Validation Loss: 1.0227 Validation Accuracy: 71.71% Validation F1: 0.4865
# =========================
# Epoch 11/50, Train Loss: 0.1966 Train Accuracy: 93.20% Validation Loss: 1.0940 Validation Accuracy: 71.34% Validation F1: 0.4885
# =========================
# Early stopping triggered    