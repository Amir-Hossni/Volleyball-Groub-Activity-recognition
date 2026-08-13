import time
from pathlib import Path
from tqdm.auto import tqdm
import torch

from utlis.metrics import calculate_metrics, save_confusion_matrix

from utlis.checkpoint import save_checkpoint
from utlis.tensorboard import (
    create_writer,
    log_metrics,
    log_confusion_matrix,
)
from utlis.early_stopping import EarlyStopping
from utlis.logger import TrainingLogger


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        adapter,
        num_classes,
        save_path,
        class_names=None,
        log_name="training",
        epochs=50,
        early_stop_patience=10,
        use_amp=True,
        grad_clip=None,
        scheduler=None
    ):
        # Accept "cuda" / "cuda:0" strings as well as torch.device
        self.device = torch.device(device) if isinstance(device, str) else device
        self.is_cuda = self.device.type == "cuda"

        # Make sure the model actually lives on the target device
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(self.device) if hasattr(criterion, "to") else criterion

        self.adapter = adapter
        self.num_classes = num_classes
        self.save_path = save_path
        self.class_names = class_names
        self.epochs = epochs
        self.grad_clip = grad_clip

        # AMP only makes sense on CUDA
        self.use_amp = bool(use_amp and self.is_cuda)

        if self.is_cuda:
            # Fixed-size inputs -> let cuDNN pick the fastest conv algorithms once
            torch.backends.cudnn.benchmark = False

        self.writer = create_writer(f"runs/{log_name}")

        # Simple epoch-level text logger (runs/{log_name}/{log_name}.log)
        self.logger = TrainingLogger(log_name=log_name)

        self.early_stopping = EarlyStopping(
            patience=early_stop_patience,
            mode="max",
        )

        self.best_acc = 0.0
        self.last_ckpt_path = Path(save_path).parent / "last_checkpoint.pth"

        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.use_amp,
        )

    def _autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
        )

    def train_one_epoch(self, loader, epoch):
        self.model.train()

        running_loss = 0.0
        correct = 0
        seen = 0

        all_predictions = []
        all_targets = []

        epoch_start = time.time()

        train_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} [train]", leave=False)
        for batch in train_bar:
            inputs, targets = self.adapter(batch)

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # set_to_none=True is cheaper than zeroing the buffers
            self.optimizer.zero_grad(set_to_none=True)

            # Forward
            with self._autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward
            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                # Gradients must be unscaled before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )

            # Optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            batch_size = targets.size(0)
            running_loss += loss_val

            # Cumulative accuracy for tqdm progress display.
            # argmax produces a non-GPU-grad tensor, so no extra detach is needed.
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            seen += batch_size

            # Store predictions and targets for epoch-level metrics
            # (CPU to avoid GPU memory accumulation; computed once at epoch end).
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

            # # Keep tensors on GPU during epoch
            # all_predictions.append(predictions)
            # all_targets.append(targets)
            
            train_bar.set_postfix(
                loss=f"{loss_val:.4f}",
                acc=f"{correct / seen * 100:.2f}%",
            )

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        metrics = calculate_metrics(
            predictions,
            targets,
            num_classes=self.num_classes,
        )

        metrics["epoch_time"] = time.time() - epoch_start

        return running_loss / max(len(loader), 1), metrics

    def _get_model_for_save(self):
        """Unwrap DataParallel/DDP to get the underlying model."""
        if isinstance(
            self.model,
            (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
        ):
            return self.model.module
        return self.model

    def _save_last_checkpoint(self, epoch, metrics=None):
        """Save a rolling last checkpoint (overwrites each epoch)."""
        save_checkpoint(
            self.last_ckpt_path,
            self._get_model_for_save(),
            self.optimizer,
            epoch,
            metrics or {},
            scaler=self.scaler,
        )

    def fit(self, train_loader, val_loader):
        from engine.evaluator import evaluate

        epoch = 0
        try:
            for epoch in range(self.epochs):

                train_loss, train_metrics = self.train_one_epoch(train_loader, epoch)

                val_bar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1}/{self.epochs} [val]",
                    leave=False,
                )

                val_loss, val_metrics, val_predictions, val_targets = evaluate(
                    self.model,
                    val_bar,
                    self.criterion,
                    self.device,
                    self.adapter,
                    self.num_classes,
                    use_amp=self.use_amp,
                    return_preds_targets=True,
                )

                log_metrics(
                    self.writer,
                    {
                        "Loss/train": train_loss,
                        "F1/train": train_metrics["f1_score"],
                        "Loss/val": val_loss,
                        "F1/val": val_metrics["f1_score"],
                        "Accuracy/train": train_metrics["accuracy"],
                        "Accuracy/val": val_metrics["accuracy"],
                    },
                    epoch,
                )

                tqdm.write(
                    f"Epoch {epoch+1}/{self.epochs}, "
                    f"Train Loss: {train_loss:.4f} "
                    f"Train Accuracy: {train_metrics['accuracy']:.2f}% "
                    f"Validation Loss: {val_loss:.4f} "
                    f"Validation Accuracy: {val_metrics['accuracy']:.2f}% "
                    f"Validation F1: {val_metrics['f1_score']:.4f}"
                )
                tqdm.write("=" * 25)

                best_model_saved = False
                if val_metrics["accuracy"] > self.best_acc:
                    self.best_acc = val_metrics["accuracy"]
                    best_model_saved = True

                    save_checkpoint(
                        self.save_path,
                        self._get_model_for_save(),
                        self.optimizer,
                        epoch,
                        {
                            "val_acc": self.best_acc,
                            "val_f1": val_metrics["f1_score"],
                        },
                        scaler=self.scaler,
                    )

                    tqdm.write(
                        f"Best model saved (Val Accuracy = {self.best_acc:.2f}%)"
                    )

                    # Build the validation confusion matrix only when a new
                    # best model is saved, so the final image corresponds to
                    # the best checkpoint. Both save it as a PNG file and log
                    # it to TensorBoard as an image.
                    # (Returns an open figure since save_path is None.)
                    cm_fig = save_confusion_matrix(
                        val_predictions,
                        val_targets,
                        class_names=self.class_names,
                        save_path=None,
                        normalize="true",
                        title=f"Validation Confusion Matrix - Epoch {epoch + 1}",
                    )

                    cm_path = Path(self.save_path).parent / "confusion_matrix_val.png"
                    cm_path.parent.mkdir(parents=True, exist_ok=True)
                    cm_fig.savefig(cm_path, dpi=150, bbox_inches="tight")
                    tqdm.write(f"Confusion matrix saved to {cm_path}")

                    log_confusion_matrix(
                        self.writer,
                        cm_fig,
                        "ConfusionMatrix/val",
                        epoch,
                    )

                # Rolling last checkpoint (overwrites each epoch)
                self._save_last_checkpoint(
                    epoch,
                    {
                        "val_acc": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1_score"],
                    },
                )

                # Early stopping monitors validation accuracy (primary metric).
                # F1 is still calculated and logged for analysis, but no longer
                # drives checkpointing or early stopping.
                early_stopped = self.early_stopping(val_metrics["accuracy"])
                if early_stopped:
                    tqdm.write("Early stopping triggered")

                # Write epoch-level summary to the text log file.
                # This happens once per epoch (never inside the batch loop).
                self.logger.log_epoch(
                    epoch=epoch + 1,
                    total_epochs=self.epochs,
                    train_loss=train_loss,
                    train_acc=train_metrics["accuracy"],
                    val_loss=val_loss,
                    val_acc=val_metrics["accuracy"],
                    val_f1=val_metrics["f1_score"],
                    epoch_time=train_metrics["epoch_time"],
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    best_model_saved=best_model_saved,
                    early_stopped=early_stopped,
                )

                if self.scheduler is not None:
                    self.scheduler.step()
                
                if early_stopped:
                    break

        except KeyboardInterrupt:
            tqdm.write(
                f"\nTraining interrupted at epoch {epoch + 1}. "
                "Saving last checkpoint..."
            )
            self._save_last_checkpoint(epoch)

        finally:
            self.writer.close()