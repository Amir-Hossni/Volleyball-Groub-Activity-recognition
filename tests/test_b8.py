import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from Models.Baseline3.model_B3 import PersonClassifierB3
from Models.Baseline5.model_B5 import PersonTemporalB5
from Models.Baseline8.model_B8 import B8TeamHierarchicalClassifier
from engine.adapters import identity_adapter

def run_tests():
    print("=== Testing Baseline 8 Architecture & Pipeline ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build Person Model
    bare_b3 = PersonClassifierB3(num_classes=9, pretrained=False)
    backbone = bare_b3.model
    backbone.fc = nn.Identity()

    person_model = PersonTemporalB5(
        backbone=backbone,
        num_classes=9,
        lstm_hidden=512,
        lstm_layers=1,
        dropout=0.2,
    )

    # 2. Build B8 Model
    model = B8TeamHierarchicalClassifier(
        person_model=person_model,
        num_classes=8,
        group_feature_dim=3000,
        hidden_dim=500,
        feature_dropout=0.0,
        use_player_proj=False,
        pool="max",
        scene_head="conv",
    ).to(device)

    print("Model instantiated successfully.")
    print(f"Team pooled dimension: {model.team_pooled_dim} (Expected: 5120)")
    assert model.team_pooled_dim == 5120, f"Expected 5120, got {model.team_pooled_dim}"

    # 3. Simulate Batch & Adapter
    B, T, P, C, H, W = 2, 9, 12, 3, 224, 224
    dummy_batch = {
        "images": torch.randn(B, T, P, C, H, W),
        "mask": torch.ones(B, T, P, dtype=torch.bool),
        "scene_label": torch.tensor([1, 4]),
        "team": torch.tensor([
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1],  # 1 missing player
        ], dtype=torch.long),
    }
    dummy_batch["mask"][1, :, 11] = False  # player 11 absent

    inputs, targets, team_mask = identity_adapter(
        dummy_batch, input_key="images", target_key="scene_label", mask="team"
    )
    print("Adapter passed.")
    print(f"inputs shape: {inputs.shape}, team_mask shape: {team_mask.shape}")
    assert team_mask.shape == (B, T, P)
    assert team_mask[1, 0, 11] == -1, "Missing player should have mask == -1"

    # 4. Forward Pass
    inputs = inputs.to(device)
    team_mask = team_mask.to(device)
    targets = targets.to(device)

    logits = model(inputs, team_mask)
    print(f"Forward output shape: {logits.shape} (Expected: ({B}, 8))")
    assert logits.shape == (B, 8)

    # 5. Backward Pass
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()
    print(f"Backward pass loss: {loss.item():.4f}")

    # 6. Unfreeze Person LSTM (Phase 2 check)
    unfrozen = model.unfreeze_person_lstm()
    print(f"Unfrozen {len(unfrozen)} parameter tensors in Person LSTM.")
    assert all(p.requires_grad for p in unfrozen)
    assert not any(p.requires_grad for p in model.person_model.backbone.parameters()), "Backbone should stay frozen!"

    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
