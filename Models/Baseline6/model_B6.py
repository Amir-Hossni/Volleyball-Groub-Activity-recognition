import torch
import torch.nn as nn
"""
    B6: Two-stage model without LSTM1

    Frozen B3 ResNet50
    Max + Mean Pooling
    Group LSTM (LSTM2)
    Feature Projection
    Conv1D Temporal Fusion
    Group Activity Classifier
    8 Group Activity Classes
    """
class B6GroupActivityClassifier(nn.Module):
    def __init__(

        self,
        backbone,
        num_classes=8,
        num_frames=9,
        hidden_dim=512,
        dropout=0.4,
    ):
        super().__init__()


        # B3 backbone
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

        # Feature dimensions
        self.player_feature_dim = 2048
        self.team_feature_dim = self.player_feature_dim * 2
        self.hidden_dim = hidden_dim


        # Feature dropout
        self.feature_dropout = nn.Dropout(dropout)

        # Group LSTM (LSTM2)
        self.lstm = nn.LSTM(
            input_size=self.team_feature_dim,   # 4096
            hidden_size=self.hidden_dim,        # 512
            num_layers=1,
            batch_first=True,
        )


        # Bypass projection
        self.project = nn.Linear(
            self.team_feature_dim,               # 4096
            self.hidden_dim,                     # 512
        )


        # Temporal fusion
        self.fused_time_steps = num_frames * 2

        self.conv_projection = nn.Sequential(

        nn.Conv1d(
            in_channels=self.hidden_dim,     # 512
            out_channels=self.hidden_dim // 2,  # 256
            kernel_size=self.fused_time_steps,  # 18
        ),

        nn.BatchNorm1d(self.hidden_dim // 2),
        nn.ReLU(inplace=True),

        nn.Conv1d(
            in_channels=self.hidden_dim // 2,  # 256
            out_channels=self.hidden_dim // 4, # 128
            kernel_size=1,
        ),

        nn.BatchNorm1d(self.hidden_dim // 4),
        nn.ReLU(inplace=True),

        nn.Flatten()
        )


        # Group classifier
        classifier_input = self.hidden_dim // 4  # 128

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, self.hidden_dim), # 512 → 128                
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2,), # 512 → 256
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_classes, ),# 256 8
        )
    

    def train(self, mode=True):

        super().train(mode)

        # Keep frozen B3 backbone in eval mode.
        self.backbone.eval()
        return self


    def forward(self, x, mask):
        """
        Args:
            x:
              (B,T,P,C,H,W)

                Example:
                (16,9,12,3,224,224)
            mask:
                (B,T,P)
                True  = real player
                False = padded/missing player
        Returns:
            logits:
                (B,8)
        """

        B, T, P, C, H, W = x.shape

        
        # 1. Flatten all player crops: (B,T,P,C,H,W) -> (B*T*P,C,H,W)
        x = x.reshape(B * T * P, C, H, W)


        # Flatten mask using the same ordering.
        flat_mask = mask.reshape(B * T * P)


        # Keep only real player crops.
        valid_x = x[flat_mask]


        
        # 2. Frozen B3 ResNet50
        # Each real player crop independently becomes:
        # (C,H,W) -> ResNet50 -> (2048)
        with torch.no_grad():

            valid_features = self.backbone(valid_x)

        valid_features = valid_features.flatten(1)

     
        # 3. Restore missing players as zero vectors: (B*T*P,2048)
        features = torch.zeros(
            B * T * P,
            2048,
            device=x.device,
            dtype=valid_features.dtype,
        )

        features[flat_mask] = valid_features

        # Restore: (B*T*P,2048) -> (B,T,P,2048)
        features = features.reshape(B, T, P, 2048)

        # 4. Feature dropout
        features = self.feature_dropout(features)

        # 5. Player pooling
        # MAX:(B,T,P,2048) -> max P : (B,T,2048)
        # MEAN: (B,T,P,2048) -> mean P : (B,T,2048)
        mask4 = mask.unsqueeze(-1)  # (B,T,P,1)
       
       
        # Max pooling
        masked_features = features.masked_fill(~mask4, float("-inf"))

        pooled_max = masked_features.max(dim=2).values

        # If a frame has no valid players,
        # replace -inf with zero.
        pooled_max = torch.where(
            torch.isinf(pooled_max),
            torch.zeros_like(pooled_max),
            pooled_max
        )


        # Mean pooling
        masked_sum = features.masked_fill(~mask4, 0.0).sum(dim=2)

        valid_counts = (
            mask.sum(dim=2, keepdim=True)
            .clamp_min(1)
            .float()
        )

        pooled_mean = masked_sum / valid_counts
  
        # 6. Combine max + mean
        team = torch.cat([pooled_max, pooled_mean], dim=-1)

        # 7. Group LSTM = LSTM 2
        out, _ = self.lstm(team)
     
        # 8. Bypass / Projection
        team_projected = self.project(team)

        # 9. Fuse LSTM information + original team information
        combined = torch.cat([out, team_projected], dim=1)
       
        # 10. Prepare for Conv1D
        combined = combined.permute(0, 2, 1)

        # 11. Temporal convolution
        clip_features = self.conv_projection(combined)

        # 12. Final group classification
        logits = self.classifier(clip_features)

        return logits 