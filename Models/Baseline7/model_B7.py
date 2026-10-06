import torch
import torch.nn as nn




class B7HierarchicalClassifier(nn.Module):
    """
        B7: Two-stage hierarchical model (full model, one group)
        Ibrahim et al., CVPR 2016 — Sec. 3.2 / 3.3, Eq. 7–8

        Stage 1 (trained before, person actions):
            PersonTemporalB5 = frozen B3 ResNet50 + LSTM1 (9 classes)

        Stage 2 (this model, group activity):

        Player crops
            ↓
        Frozen B3 ResNet50            x_t^k : 2048
            ↓
        LSTM1 over time, per player   h_t^k : H1 (512)
            ↓
        P_t^k = x_t^k ⊕ h_t^k         2048 + H1 = 2560
            ↓
        Max Pooling over Players      Z_t   : 2560
            ↓
        Group Feature Projection: 2560 → 3000
            ↓
        Group LSTM (LSTM2): 3000 → 500
            ↓
        Group Activity Classifier
            ↓
        8 Group Activity Classes

        Paper Eq. 7: P_t^k = x_t^k ⊕ h_t^k
        Paper Eq. 8: Z_t = max over K players

        LSTM1 starts frozen (paper: stage-wise training).
        unfreeze_person_lstm() opens it for an optional joint fine-tune
        with a lower learning rate.
    """
    def __init__(

        self,
        person_model,
        num_classes=8,
        group_feature_dim=3000,
        hidden_dim=500,
        feature_dropout=0.0,
    ):
        super().__init__()


        # Stage-1 person model (PersonTemporalB5): frozen B3 backbone + LSTM1
        self.person_model = person_model

        for param in self.person_model.parameters():
            param.requires_grad = False

        self.person_model.eval()
        self.person_trainable = False


        # Feature dimensions
        self.player_feature_dim = 2048
        self.person_hidden_dim = self.person_model.lstm.hidden_size        # H1
        self.player_repr_dim = self.player_feature_dim + self.person_hidden_dim
        self.group_feature_dim = group_feature_dim
        self.hidden_dim = hidden_dim


        # Dropout on the frozen CNN features (0 = off)
        self.feature_dropout = nn.Dropout(feature_dropout)


        # Normalize the pooled group feature Z_t before the FC 3000.
        # Not in the paper: raw pooled ResNet features saturated the LSTM2
        # gates and B8 did not train without it; with it B7/B8 train stably.

        self.group_norm = nn.LayerNorm(self.player_repr_dim)       # 2560


        # Group feature projection
        # Paper: 3000-node fully connected layer

        self.group_projection = nn.Linear(
            self.player_repr_dim,        # 2560
            self.group_feature_dim,      # 3000
        )


        # Group LSTM (LSTM2)
        # Paper: 9 timesteps, 500 hidden nodes

        self.lstm = nn.LSTM(
            input_size=self.group_feature_dim,   # 3000
            hidden_size=self.hidden_dim,         # 500
            num_layers=1,
            batch_first=True,
        )


        # Group classifier

        classifier_input = self.hidden_dim  # 500

        # Paper: "h_t^group is fed to a softmax classification layer"
        self.classifier = nn.Linear(classifier_input, num_classes)  # 500 → 8


    def unfreeze_person_lstm(self):
        """
        Open LSTM1 for joint fine-tuning.
        Returns its parameters for a low-LR optimizer group.
        The ResNet backbone and the 9-class action head stay frozen.
        """
        params = list(self.person_model.lstm.parameters())

        for param in params:
            param.requires_grad = True

        self.person_trainable = True

        return params


    def train(self, mode=True):

        super().train(mode)

        # Frozen parts stay in eval mode.
        self.person_model.backbone.eval()

        if not self.person_trainable:
            self.person_model.eval()

        return self


    def player_representations(self, x, mask):
        """
        Args:
            x    : (B,T,P,C,H,W)
            mask : (B,T,P)  True = real player

        Returns:
            P_t^k = x_t^k ⊕ h_t^k : (B,T,P,2048+H1)
        """

        B, T, P, C, H, W = x.shape


        #  Frozen B3 ResNet50 on real crops only
        # (B*T*P,C,H,W) -> (N_valid,2048)

        flat_x = x.reshape(B * T * P, C, H, W)
        flat_mask = mask.reshape(B * T * P)

        with torch.no_grad():

            valid_features = self.person_model.backbone(
                flat_x[flat_mask]
            ).flatten(1)


        #  Restore missing crops as zero vectors
        # (B*T*P,2048) -> (B,T,P,2048)

        features = torch.zeros(
            B * T * P,
            self.player_feature_dim,
            device=x.device,
            dtype=valid_features.dtype,
        )

        features[flat_mask] = valid_features

        features = features.reshape(B, T, P, self.player_feature_dim)

        features = self.feature_dropout(features)


        #  LSTM1 over time, one track per player
        # (B,T,P,2048) -> (B*P,T,2048)

        tracks = features.permute(0, 2, 1, 3).reshape(B * P, T, -1)

        # Players absent from the whole clip are skipped
        track_valid = mask.any(dim=1).reshape(B * P)

        hidden = torch.zeros(
            B * P,
            T,
            self.person_hidden_dim,
            device=x.device,
            dtype=features.dtype,
        )

        with torch.set_grad_enabled(self.person_trainable and torch.is_grad_enabled()):

            track_out, _ = self.person_model.lstm(tracks[track_valid])

        hidden[track_valid] = track_out.to(hidden.dtype)


        # (B*P,T,H1) -> (B,T,P,H1)

        hidden = hidden.reshape(B, P, T, -1).permute(0, 2, 1, 3)


        #  Paper Eq. 7: P_t^k = x_t^k ⊕ h_t^k

        return torch.cat([features, hidden], dim=-1)


    @staticmethod
    def masked_max(player_repr, mask):
        """
        Paper Eq. 8: max pooling over the selected players.

        (B,T,P,D) + (B,T,P) -> (B,T,D)
        A frame with no selected player gives a zero vector.
        """

        pooled = player_repr.masked_fill(
            ~mask.unsqueeze(-1),
            float("-inf")
        ).max(dim=2).values

        return torch.where(
            torch.isinf(pooled),
            torch.zeros_like(pooled),
            pooled
        )


    def classify_group(self, group_features):
        """
        (B,T,D) -> LayerNorm -> FC 3000 -> LSTM2 500 -> last step -> (B,8)
        """

        group_features = self.group_norm(group_features)

        projected = self.group_projection(group_features)   # (B,T,3000)

        out, _ = self.lstm(projected)                        # (B,T,500)

        return self.classifier(out[:, -1, :])               # (B,8)


    def forward(self, x, mask):
        """
        Args:
            x:
              (B,T,P,C,H,W)

                Example:
                (8,9,12,3,224,224)

            mask:
                (B,T,P)
                True  = real player
                False = padded/missing player

        Returns:
            logits:
                (B,8)
        """

        player_repr = self.player_representations(x, mask)   # (B,T,P,2560)

        group_features = self.masked_max(player_repr, mask)  # (B,T,2560)

        return self.classify_group(group_features)           # (B,8)
