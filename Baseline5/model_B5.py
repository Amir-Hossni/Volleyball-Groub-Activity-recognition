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
    

class GroupTemporalClassifierB5(nn.Module):

    def __init__(
        self,
        person_model,
        num_classes=8,
        hidden_dim=4096,
        dropout=0.2
    ):
        super().__init__()

        # Stage-A temporal person model
        self.person_model = person_model

        # Freeze Stage-A completely
        for param in self.person_model.parameters():
            param.requires_grad = False

        # Same classifier idea as B3
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def train(self, mode=True):
        # Train Stage-B classifier
        super().train(mode)

        # Keep the frozen Stage-A model in evaluation mode
        # so BatchNorm running statistics do not change.
        self.person_model.eval()

        return self
    
    
    
    def forward(self, x):

        # x: (B, P, T, C, H, W)
        B, P, T, C, H, W = x.shape

        # Merge batch and player dimensions
        x = x.reshape(B * P, T, C, H, W)

        # Stage-A temporal representation
        with torch.no_grad():
            player_features = self.person_model(x, return_features=True) # (B*P, 512)

        
        # Restore player dimension
        player_features = player_features.reshape(B, P, -1 ) # (B, 12, 512)

        # Max pooling over players
        team_features, _ = torch.max(player_features, dim=1) # (B, 512)

        output = self.classifier(team_features) # (B, 8)

        return output



class GroupTemporalClassifierB5V2(nn.Module):

    def __init__(
        self,
        backbone,
        lstm,
        num_classes=8,
        hidden_dim=4096,
        dropout=0.2
    ):
        super().__init__()

        # B5 Stage-1 components
        self.backbone = backbone
        self.lstm = lstm

        # Remove ResNet classification head
        self.backbone.fc = nn.Identity()

        # Freeze Stage-1
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.lstm.parameters():
            param.requires_grad = False

        # Stage-2 classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),

            nn.Linear(2048, num_classes)
        )

    def train(self, mode=True):

        super().train(mode)

        # Keep frozen Stage-1 in evaluation mode
        self.backbone.eval()
        self.lstm.eval()

        return self

    def forward(self, x):

        # x: (B, P, T, C, H, W)

        B, P, T, C, H, W = x.shape

        # (B*P, T, C, H, W)
        x = x.reshape(B * P, T, C, H, W)

        # (B*P*T, C, H, W)
        x = x.reshape(B * P * T, C, H, W)

        # -----------------------------------------
        # Stage-1 CNN
        # -----------------------------------------

        with torch.no_grad():
            features = self.backbone(x)

        # (B*P*T, 2048)
        features = features.reshape(
            B * P,
            T,
            2048
        )

        # -----------------------------------------
        # Stage-1 LSTM
        # -----------------------------------------

        with torch.no_grad():
            _, (h_n, _) = self.lstm(features)

        # (B*P, 512)
        player_features = h_n[-1]

        # (B, P, 512)
        player_features = player_features.reshape(
            B,
            P,
            512
        )

        # -----------------------------------------
        # Max pooling over players
        # -----------------------------------------

        team_features, _ = torch.max(
            player_features,
            dim=1
        )

        # (B, 512)
        output = self.classifier(team_features)

        # (B, 8)
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