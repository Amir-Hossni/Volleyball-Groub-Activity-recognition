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
        P_t^k : raw concat 2048 + H1 (paper Eq. 7, default)
                or projected concat H1 + H1 with use_player_proj
            ↓
        Pooling over all Players (one group, paper Eq. 8):
            max (default) or max||mean with pool="max_mean"
            ↓
        Group Feature Projection: pooled_dim → 3000
            ↓
        Group LSTM (LSTM2): 3000 → 500
            ↓
        Scene head: last (paper) / mean / conv (Omar-style)
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
        use_player_proj=False,
        pool="max",
        scene_head="last",
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

        # Player representation size
        # B7 default: raw concat (paper Eq. 7)
        #   P = x (2048) + h (H1)
        # With player projection (Omar-style, B7-compatible):
        #   P = proj(x) (H1) + h (H1) = 2 * H1
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


        # Player pooling over all players (one group, paper Eq. 8)
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
        # "last" : paper head on LSTM2 last step        -> 500
        # "mean" : mean over LSTM2 steps                -> 500
        # "conv" : Omar-style time concat + Conv1d      -> hidden_dim // 4
        if scene_head not in ("last", "mean", "conv"):
            raise ValueError("scene_head must be 'last', 'mean' or 'conv'")

        self.scene_head = scene_head

        if scene_head == "conv":
            # Scene skip: project pooled Z_t to LSTM width, concat with
            # LSTM2 outputs on the TIME axis -> (B, 2T, H2), then a
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
        self.classifier = nn.Linear(classifier_input, num_classes)


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


        if self.use_player_proj:
            # Omar-style player skip: project raw CNN features to H1,
            # then concat with LSTM1 output -> (B,T,P,2*H1)
            with torch.set_grad_enabled(self.person_trainable and torch.is_grad_enabled()):
                proj_in = features.reshape(B * T * P, self.player_feature_dim)
                proj_out = self.player_proj(proj_in).reshape(B, T, P, -1)

            return torch.cat([proj_out, hidden], dim=-1)

        # Paper Eq. 7: P_t^k = x_t^k ⊕ h_t^k

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


    @staticmethod
    def masked_mean(player_repr, mask):
        """
        Mean pooling over the selected players.

        (B,T,P,D) + (B,T,P) -> (B,T,D)
        A frame with no selected player gives a zero vector.
        """

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
        """
        Pool over all players (one group, no team split).

        "max"      -> (B,T,D)
        "max_mean" -> (B,T,2*D)  (max || mean)
        """

        pooled_max = self.masked_max(player_repr, mask)

        if self.pool == "max":
            return pooled_max

        pooled_mean = self.masked_mean(player_repr, mask)

        return torch.cat([pooled_max, pooled_mean], dim=-1)


    def classify_group(self, group_features):
        """
        (B,T,D) -> LayerNorm -> FC 3000 -> LSTM2 500 -> head -> (B,8)

        "last" : paper head on LSTM2 last step
        "mean" : mean over LSTM2 steps
        "conv" : Omar-style time concat + Conv1d fusion
        """

        group_features = self.group_norm(group_features)

        projected = self.group_projection(group_features)   # (B,T,3000)

        out, _ = self.lstm(projected)                        # (B,T,500)

        if self.scene_head == "last":
            return self.classifier(out[:, -1, :])            # (B,8)

        if self.scene_head == "mean":
            return self.classifier(out.mean(dim=1))          # (B,8)

        # Omar-style scene fusion:
        # LSTM2 outputs (B,T,500) || projected scene (B,T,500)
        #   -> (B,2T,500) -> Conv1d -> (B,125) -> classifier
        skip = self.scene_skip_proj(projected)               # (B,T,500)

        fused = torch.cat([out, skip], dim=1)                # (B,2T,500)
        fused = fused.permute(0, 2, 1)                       # (B,500,2T)

        fused = torch.relu(self.scene_conv1(fused))          # (B,250,2T)
        fused = self.scene_conv2(fused)                      # (B,125,1)

        return self.classifier(fused.squeeze(-1))            # (B,8)


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

        player_repr = self.player_representations(x, mask)

        group_features = self.pool_players(player_repr, mask)

        return self.classify_group(group_features)           # (B,8)
