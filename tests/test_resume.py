import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from engine.trainer import Trainer

def test_resume():
    print("=== Testing Trainer Resume Functionality ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 8)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    ckpt_file = Path("test_ckpt.pth")
    try:
        # Create a mock checkpoint saved at epoch 12 with val_acc 72.5
        mock_ckpt = {
            "epoch": 12,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": 72.5,
        }
        torch.save(mock_ckpt, ckpt_file)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            adapter=lambda b: (b[0], b[1]),
            num_classes=8,
            save_path="test_save.pth",
            epochs=35,
        )

        assert trainer.start_epoch == 0
        assert trainer.best_acc == 0.0

        # Resume from checkpoint
        trainer.resume(str(ckpt_file))

        print(f"Resumed start_epoch: {trainer.start_epoch} (Expected: 13)")
        print(f"Resumed best_acc: {trainer.best_acc} (Expected: 72.5)")
        assert trainer.start_epoch == 13, f"Expected 13, got {trainer.start_epoch}"
        assert trainer.best_acc == 72.5, f"Expected 72.5, got {trainer.best_acc}"
        assert trainer.early_stopping.best_score == 72.5

        print("RESUME TEST PASSED SUCCESSFULLY!")

    finally:
        if ckpt_file.exists():
            ckpt_file.unlink()

if __name__ == "__main__":
    test_resume()

