import torch
import torch.nn as nn


class B6GroupActivityClassifier(nn.Module):
    """

    Each player crop from every frame is passed independently through
    the frozen B3 ResNet50 backbone.

    Then, player-level features are aggregated using max pooling
    across the player dimension:

    This produces one group-level feature vector for each frame.

    The sequence of 9 frame-level features is then processed by a
    group-level LSTM:

    Finally, a classifier predicts the 8 group activity classes.

    Overall flow:

        B3 Stage 1
        Single player crop
              ↓
        Fine-tuned ResNet50
              ↓
        Save / load B3 checkpoint
              ↓
        Freeze backbone
              ↓
        B6: 9 frames × 12 players
              ↓
        B3 ResNet50 features
              ↓
        Max Pool across players
              ↓
        9 frame-level features
              ↓
        Group LSTM
              ↓
        8 Group Activity classes
    """

    def __init__(
        self,
        backbone,
        hidden_size=512,
        num_classes=8,
        dropout=0.4
    ):
        super().__init__()


        # Stage 1: Reuse B3 Stage 1 ResNet50 backbone
        self.backbone = backbone

        # Remove B3's person-action classification head.
        # We only need the 2048-dimensional feature representation.
        self.backbone.fc = nn.Identity()

        # Freeze the entire B3 backbone during B6 training.
        for param in self.backbone.parameters():
            param.requires_grad = False


        # Stage 2: Group-level temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Final group-activity classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
            nn.Linear(128 ,num_classes)
        )

    def train(self, mode=True):
            super().train(mode)
            # Keep the frozen backbone in eval mode no matter what mode
            # the parent model is switched to.
            self.backbone.eval()
            return self
    
    def forward(self, x, mask):
        """
        Args:
            x:
                Tensor with shape:
                    (B, T, P, C, H, W)
                Example:
                    (16, 9, 12, 3, 224, 224)
        Returns:
            logits:
                Tensor with shape:
                    (B, 8)
        """

        B, T, P, C, H, W = x.shape
        # Apply the frozen B3 backbone independently to every
        # player crop in every frame.
        #
        # (B,T,P,C,H,W)
        #        ↓
        # (B*T*P,C,H,W)


        x = x.reshape(B * T * P, C, H, W)
        
        # Flatten mask in the same order
        flat_mask = mask.reshape(B * T * P)

        # Select only real players
        valid_x = x[flat_mask]
        
        # No gradients are required because the backbone
        # is frozen.
        with torch.no_grad():
            valid_features = self.backbone(valid_x)
        
        valid_features = valid_features.flatten(1)

     
        # 4. Create feature tensor for all players
        #    Missing/padded players remain zero
        features = torch.zeros(
            B * T * P,
            2048,
            device=x.device,
            dtype=valid_features.dtype,
        )

        features[flat_mask] = valid_features


        # Restore the temporal and player dimensions
        features = features.reshape(B, T, P, 2048) # (B*T*P,2048) -> (B,T,P,2048)

    
        # Max pooling across players
        pooled_features = features.max(dim=2).values # (B,T,P,2048) -> (B,T,2048)


        # Group-level temporal modeling
        lstm_out, _ = self.lstm(pooled_features) # (B,T,2048)  ->  Group LSTM -> (B,T,hidden_size)

        # Use the representation of the last frame
        # as the representation of the whole 9-frame clip.
        clip_features = lstm_out[:, -1, :]

        # Group activity classification
        logits = self.classifier(clip_features) # (B,hidden_size) -> (B,8)

        return logits