import time

import torch

from utlis.metrics import calculate_metrics
from utlis.checkpoint import save_checkpoint
from utlis.tensorboard import create_writer, log_metrics
from utlis.early_stopping import EarlyStopping


class New_Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        adapter,
        num_classes,
        save_path,
        log_name="training",
        epochs=50,
        early_stop_patience=10,
        use_amp=True,
        grad_clip=None,
    ):
        # Accept "cuda" / "cuda:0" strings as well as torch.device
        self.device = torch.device(device) if isinstance(device, str) else device
        self.is_cuda = self.device.type == "cuda"

        # Make sure the model actually lives on the target device
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.criterion = criterion.to(self.device) if hasattr(criterion, "to") else criterion

        self.adapter = adapter
        self.num_classes = num_classes
        self.save_path = save_path
        self.epochs = epochs
        self.grad_clip = grad_clip

        # AMP only makes sense on CUDA
        self.use_amp = bool(use_amp and self.is_cuda)

        if self.is_cuda:
            # Fixed-size inputs -> let cuDNN pick the fastest conv algorithms once
            torch.backends.cudnn.benchmark = True

        self.writer = create_writer(f"runs/{log_name}")

        self.early_stopping = EarlyStopping(
            patience=early_stop_patience,
            mode="max",
        )

        self.best_f1 = 0.0

        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.use_amp,
        )

    def _autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
        )

    def train_one_epoch(self, loader):
        self.model.train()

        running_loss = 0.0
        num_samples = 0

        all_predictions = []
        all_targets = []

        epoch_start = time.time()

        for batch in loader:
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

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            # Metrics: detach and move off the GPU so VRAM doesn't grow with the epoch
            predictions = outputs.detach().argmax(dim=1)
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.detach().cpu())

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        metrics = calculate_metrics(
            predictions,
            targets,
            num_classes=self.num_classes,
        )

        metrics["epoch_time"] = time.time() - epoch_start

        return running_loss / max(num_samples, 1), metrics

    def fit(self, train_loader, val_loader):
        from engine.evaluator import evaluate

        for epoch in range(self.epochs):

            print("=" * 50)
            print(f"Epoch [{epoch + 1}/{self.epochs}]")
            print("=" * 50)

            train_loss, train_metrics = self.train_one_epoch(train_loader)

            val_loss, val_metrics = evaluate(
                self.model,
                val_loader,
                self.criterion,
                self.device,
                self.adapter,
                self.num_classes,
            )

            log_metrics(
                self.writer,
                {
                    "Loss/train": train_loss,
                    "F1/train": train_metrics["f1_score"],
                    "Loss/val": val_loss,
                    "F1/val": val_metrics["f1_score"],
                },
                epoch,
            )

            print(
                f"""
Train Loss: {train_loss:.4f}
Train F1: {train_metrics['f1_score']:.4f}

Val Loss: {val_loss:.4f}
Val F1: {val_metrics['f1_score']:.4f}

Epoch time: {train_metrics['epoch_time']:.1f}s
"""
            )

            if val_metrics["f1_score"] > self.best_f1:
                self.best_f1 = val_metrics["f1_score"]

                model_to_save = (
                    self.model.module
                    if isinstance(
                        self.model,
                        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
                    )
                    else self.model
                )

                save_checkpoint(
                    self.save_path,
                    model_to_save,
                    self.optimizer,
                    epoch,
                    {"val_f1": self.best_f1},
                )

                print("Best model saved")

            if self.early_stopping(val_metrics["f1_score"]):
                print("Early stopping triggered")
                break

        self.writer.close()