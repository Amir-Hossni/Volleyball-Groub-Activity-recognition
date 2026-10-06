import torch
import torch.nn as nn

from Models.Baseline7.model_B7 import B7HierarchicalClassifier




class B8TeamHierarchicalClassifier(B7HierarchicalClassifier):
    """
        B8: Two-stage hierarchical model with team pooling (two groups)
        Ibrahim et al., journal extension (arXiv:1607.02643)

        Same as B7, except the pooling step:
        instead of one max over all 12 players, each team is pooled
        separately and the two team vectors are concatenated.
        This keeps "which side did what" — the l_* / r_* distinction
        that pooling all players together erases.

        Player crops
            ↓
        Frozen B3 ResNet50 + LSTM1      P_t^k : 2048 + H1 = 2560
            ↓
        Max over left team  ⊕  Max over right team
            ↓
        Z_t : 2 × 2560 = 5120
            ↓
        LayerNorm (added for stability, not in the paper)
            ↓
        Group Projection: FC 3000
            ↓
        Group LSTM (LSTM2): 500
            ↓
        Linear 500 → 8 (softmax)

        Teams come from the dataset ("team": players ranked by box
        center-x, left half = 0, right half = 1) and reach the model through
        identity_adapter(mask="team") as one (B,T,P) tensor:
            0 / 1 = team of a real player,  -1 = missing player
    """
    def __init__(
        self,
        person_model,
        num_classes=8,
        group_feature_dim=3000,
        hidden_dim=500,
        feature_dropout=0.0,
    ):
        super().__init__(
            person_model=person_model,
            num_classes=num_classes,
            group_feature_dim=group_feature_dim,
            hidden_dim=hidden_dim,
            feature_dropout=feature_dropout,
        )


        # Two teams concatenated -> group feature doubles

        self.group_norm = nn.LayerNorm(2 * self.player_repr_dim)   # 5120

        # Paper: 3000-node fully connected layer

        self.group_projection = nn.Linear(
            2 * self.player_repr_dim,    # 5120
            self.group_feature_dim,      # 3000
        )


    def forward(self, x, team_mask):
        """
        Args:
            x:
              (B,T,P,C,H,W)

            team_mask:
                (B,T,P)
                0  = left-team player
                1  = right-team player
                -1 = padded/missing player

        Returns:
            logits:
                (B,8)
        """

        mask = team_mask >= 0                                     # (B,T,P)

        player_repr = self.player_representations(x, mask)       # (B,T,P,2560)


        #  Team pooling per frame
        # (B,T,P,2560) -> left (B,T,2560) ⊕ right (B,T,2560)

        left = self.masked_max(player_repr, mask & (team_mask == 0))
        right = self.masked_max(player_repr, mask & (team_mask == 1))

        group_features = torch.cat([left, right], dim=-1)        # (B,T,5120)

        return self.classify_group(group_features)               # (B,8)
