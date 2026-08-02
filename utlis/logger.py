from pathlib import Path


class TrainingLogger:
    """Simple epoch-level training logger that writes to a text file.

    Only epoch-level information is recorded (never per-batch) so that
    training speed is not affected. The log file is opened in append mode
    so that resuming training continues writing to the same file.

    Log file location:
        runs/{log_name}/{log_name}.log
    """

    def __init__(self, log_name="training", log_dir="runs"):
        # runs/{log_name}/{log_name}.log
        self.log_dir = Path(log_dir) / log_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{log_name}.log"

    def log_epoch(
        self,
        epoch,
        total_epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        val_f1,
        epoch_time,
        learning_rate,
        best_model_saved=False,
        early_stopped=False,
    ):
        """Append one epoch's summary to the log file."""
        lines = [
            f"Epoch {epoch}/{total_epochs}",
            f"Train Loss: {train_loss:.4f}",
            f"Train Accuracy: {train_acc:.2f}%",
            f"Validation Loss: {val_loss:.4f}",
            f"Validation Accuracy: {val_acc:.2f}%",
            f"Validation F1: {val_f1:.4f}",
            f"Learning Rate: {learning_rate:.6f}",
            f"Epoch Time: {epoch_time:.1f} seconds",
            f"Best Model Saved: {'Yes' if best_model_saved else 'No'}",
            f"Early Stopping: {'Yes' if early_stopped else 'No'}",
            "-" * 32,
        ]

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")