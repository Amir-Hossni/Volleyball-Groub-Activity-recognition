import torch
import torch.nn as nn




class B7HierarchicalClassifier(nn.Module):
    """
        B7: Two-stage hierarchical model (full model, one group)

        Frozen B3 ResNet50 + LSTM1 (Stage 1, person actions)
            ↓
        Player Representations: 2048 + H1 (raw) or H1 + H1 (projected)
            ↓
        Pooling over Players (one group)
            ↓
        Group Feature: pooled_dim
            ↓
        Group Feature Projection: pooled_dim → 3000
            ↓
        Group LSTM (LSTM2): 3000 → 500
            ↓
        Group Activity Classifier
            ↓
        8 Group Activity Classes
    """
    def __init__(

        self,
        person_model,
        num_classes=8,
        group_feature_dim=3000,
        hidden_dim=500,
        feature_dropout=0.0,
        use_player_proj=False,
        pool="max",
        scene_head="last",
        backbone_chunk=256,
    ):
        super().__init__()


        # Stage-1 person model (PersonTemporalB5): frozen B3 backbone + LSTM1
        self.person_model = person_model

        for param in self.person_model.parameters():
            param.requires_grad = False

        self.person_model.eval()
        self.person_trainable = False


        # Chunked frozen-backbone inference.
        self.backbone_chunk = int(backbone_chunk)


        # Feature dimensions
        self.player_feature_dim = 2048
        self.person_hidden_dim = self.person_model.lstm.hidden_size        # H1


        # Player representation size
        #   raw concat: x (2048) + h (H1)
        #   projected : proj(x) (H1) + h (H1) = 2 * H1
        self.use_player_proj = use_player_proj
        if self.use_player_proj:
            self.player_proj = nn.Linear(
                self.player_feature_dim,     # 2048
                self.person_hidden_dim,      # H1 (512)
            )
            self.player_repr_dim = 2 * self.person_hidden_dim   # 1024
        else:
            self.player_proj = None
            self.player_repr_dim = self.player_feature_dim + self.person_hidden_dim  # 2560

        self.group_feature_dim = group_feature_dim
        self.hidden_dim = hidden_dim


        # Dropout on the frozen CNN features (0 = off)
        self.feature_dropout = nn.Dropout(feature_dropout)


        # Player pooling over all players (one group)
        # "max"         -> D
        # "max_mean"    -> 2 * D  (max || mean, no team split)
        if pool not in ("max", "max_mean"):
            raise ValueError("pool must be 'max' or 'max_mean'")

        self.pool = pool
        self.pooled_dim = (
            self.player_repr_dim
            if pool == "max"
            else 2 * self.player_repr_dim
        )


        # Normalize the pooled group feature Z_t before the FC 3000.
        # Not in the paper: raw pooled ResNet features saturated the LSTM2
        # gates and B8 did not train without it; with it B7/B8 train stably.

        self.group_norm = nn.LayerNorm(self.pooled_dim)


        # Group feature projection
        # Paper: 3000-node fully connected layer

        self.group_projection = nn.Linear(
            self.pooled_dim,
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
        # "last" : head on LSTM2 last step   -> 500
        # "mean" : head on LSTM2 mean        -> 500
        # "conv" : time concat + Conv1d      -> hidden_dim // 4
        if scene_head not in ("last", "mean", "conv"):
            raise ValueError("scene_head must be 'last', 'mean' or 'conv'")

        self.scene_head = scene_head

        if scene_head == "conv":
            # Scene head: project pooled features to LSTM width, concat
            # with LSTM2 outputs on the TIME axis -> (B, 2T, H2), then a
            # two-stage Conv1d with global kernel 2T collapses time.
            self.scene_skip_proj = nn.Linear(
                self.group_feature_dim,      # 3000
                self.hidden_dim,             # 500
            )
            self.scene_conv1 = nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim // 2,
                kernel_size=1,
            )
            self.scene_conv2 = nn.Conv1d(
                in_channels=self.hidden_dim // 2,
                out_channels=self.hidden_dim // 4,
                kernel_size=18,              # 2 * T with T = 9
            )
            classifier_input = self.hidden_dim // 4    # 125
        else:
            self.scene_skip_proj = None
            self.scene_conv1 = None
            self.scene_conv2 = None
            classifier_input = self.hidden_dim         # 500

        # Paper: "h_t^group is fed to a softmax classification layer"
        self.classifier = nn.Linear(classifier_input, num_classes)  # 500 → 8


    def train(self, mode=True):

        super().train(mode)

        # Keep frozen B3 backbone in eval mode.
        self.person_model.backbone.eval()

        if not self.person_trainable:
            self.person_model.eval()

        return self


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


    def backbone_features(self, crops):
        #  Frozen backbone in chunks:
        # (N,C,H,W) -> (N,2048), order preserved
        feats = []
        chunk = max(1, self.backbone_chunk)
        with torch.no_grad():
            for i in range(0, crops.shape[0], chunk):
                feats.append(
                    self.person_model.backbone(
                        crops[i:i + chunk]
                    ).flatten(1)
                )
        return torch.cat(feats, dim=0) if feats else None


    def player_representations(self, x, mask):
        #  Player representations:
        # (B,T,P,C,H,W) + (B,T,P) -> (B,T,P,D)

        B, T, P, C, H, W = x.shape


        #  Flatten all player crops:
        # (B,T,P,C,H,W) -> (B*T*P,C,H,W)

        flat_x = x.reshape(B * T * P, C, H, W)
        flat_mask = mask.reshape(B * T * P)


        # Keep only real player crops.

        valid_x = flat_x[flat_mask]

        if valid_x.shape[0] == 0:
            valid_features = torch.zeros(
                0, self.player_feature_dim, device=x.device
            )
        else:
            valid_features = self.backbone_features(valid_x)


        #  Restore missing crops as zero vectors:
        # (B*T*P,2048) -> (B,T,P,2048)

        features = torch.zeros(
            B * T * P,
            self.player_feature_dim,
            device=x.device,
            dtype=valid_features.dtype,
        )

        features[flat_mask] = valid_features

        features = features.reshape(B, T, P, self.player_feature_dim)


        # Feature dropout

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


        #  Player representations:
        # raw concat: x ⊕ h  /  projected: proj(x) ⊕ h

        if self.use_player_proj:
            proj_in = features.reshape(B * T * P, self.player_feature_dim)
            with torch.set_grad_enabled(self.person_trainable and torch.is_grad_enabled()):
                proj_out = self.player_proj(proj_in).reshape(B, T, P, -1)

            return torch.cat([proj_out, hidden], dim=-1)

        return torch.cat([features, hidden], dim=-1)

        # (B,T,P,D)


    @staticmethod
    def masked_max(player_repr, mask):
        #  Player Max Pooling
        #
        # (B,T,P,D)
        #        ↓ max over P
        # (B,T,D)

        pooled = player_repr.masked_fill(
            ~mask.unsqueeze(-1),
            float("-inf")
        ).max(dim=2).values


        # If a frame has no valid players,
        # replace -inf with zero.

        return torch.where(
            torch.isinf(pooled),
            torch.zeros_like(pooled),
            pooled
        )


    @staticmethod
    def masked_mean(player_repr, mask):
        #  Player Mean Pooling
        #
        # (B,T,P,D)
        #        ↓ mean over P
        # (B,T,D)

        masked = player_repr.masked_fill(
            ~mask.unsqueeze(-1),
            0.0,
        ).sum(dim=2)

        counts = (
            mask.sum(dim=2, keepdim=True)
            .clamp_min(1)
            .float()
        )

        return masked / counts


    def pool_players(self, player_repr, mask):
        #  Player Pooling (one group)
        #
        # "max"      -> (B,T,D)
        # "max_mean" -> (B,T,2*D)  (max || mean)

        pooled_max = self.masked_max(player_repr, mask)

        if self.pool == "max":
            return pooled_max

        pooled_mean = self.masked_mean(player_repr, mask)

        return torch.cat([pooled_max, pooled_mean], dim=-1)


    def classify_group(self, group_features):
        #  Group feature projection
        #
        # (B,T,pooled_dim)
        #      ↓
        # (B,T,3000)

        group_features = self.group_norm(group_features)

        projected = self.group_projection(group_features)   # (B,T,3000)


        #  Group LSTM = LSTM 2
        #
        # (B,T,3000)
        #      ↓
        # (B,T,500)

        out, _ = self.lstm(projected)                        # (B,T,500)


        #  Final group classification

        if self.scene_head == "last":

            # Use final group LSTM representation.

            final_feature = out[:, -1, :]                    # (B,500)

            logits = self.classifier(final_feature)          # (B,8)

            return logits

        if self.scene_head == "mean":

            # Use mean group LSTM representation.

            mean_feature = out.mean(dim=1)                    # (B,500)

            logits = self.classifier(mean_feature)           # (B,8)

            return logits


        # Scene head fusion:
        # LSTM2 outputs (B,T,500) || projected scene (B,T,500)
        #   -> (B,2T,500) -> Conv1d -> (B,125) -> classifier
        skip = self.scene_skip_proj(projected)               # (B,T,500)

        fused = torch.cat([out, skip], dim=1)                # (B,2T,500)
        fused = fused.permute(0, 2, 1)                       # (B,500,2T)

        fused = torch.relu(self.scene_conv1(fused))          # (B,250,2T)
        fused = self.scene_conv2(fused)                      # (B,125,1)

        logits = self.classifier(fused.squeeze(-1))          # (B,8)

        return logits


    def forward(self, x, mask):
        """
        Args:
            x:
              (B,T,P,C,H,W)

                Example:
                (4,9,12,3,224,224)

            mask:
                (B,T,P)
                True  = real player
                False = padded/missing player

        Returns:
            logits:
                (B,8)
        """


        B, T, P, C, H, W = x.shape


        #  Flatten all player crops:
        # (B,T,P,C,H,W) -> (B*T*P,C,H,W)

        flat_x = x.reshape(
            B * T * P,
            C,
            H,
            W
        )


        # Flatten mask using the same ordering.

        flat_mask = mask.reshape(
            B * T * P
        )


        # Keep only real player crops.

        valid_x = flat_x[flat_mask]


        #  Frozen B3 ResNet50 + LSTM1
        #
        # Each real player crop independently becomes (2048),
        # then LSTM1 runs one track per player over time,
        # then concat into player representations:
        #
        # (B,T,P,D)

        if valid_x.shape[0] == 0:
            player_repr = torch.zeros(
                B, T, P, self.player_repr_dim, device=x.device
            )
        else:
            player_repr = self.player_representations(x, mask)


        #  Player Pooling (one group)
        #
        # (B,T,P,D)
        #        ↓ pool over P
        # (B,T,pooled_dim)

        group_features = self.pool_players(player_repr, mask)


        #  Group feature projection + LSTM2 + classifier
        #
        # (B,T,pooled_dim)
        #      ↓
        # (B,8)

        logits = self.classify_group(group_features)

        # (B,8)


        return logits
