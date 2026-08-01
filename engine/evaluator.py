import torch

from utlis.metrics import calculate_metrics




@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    adapter,
    num_classes,
    use_amp=False,
):
    """
    Evaluate the model on the given loader.

    Args:
        model: The model to evaluate.
        loader: DataLoader or tqdm-wrapped DataLoader.
        criterion: Loss function.
        device: torch.device to run on.
        adapter: Batch adapter function.
        num_classes: Number of classes for metrics.
        use_amp: Whether to use automatic mixed precision.
    """

    model.eval()

    total_loss = 0
    correct = 0
    seen = 0

    all_predictions = []
    all_targets = []

    for batch in loader:

        inputs, targets = adapter(batch)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss_val = loss.item()
        total_loss += loss_val

        predictions = torch.argmax(outputs, dim=1)

        batch_correct = (predictions == targets).sum().item()
        correct += batch_correct
        seen += targets.size(0)

        # Move to CPU immediately to avoid GPU memory accumulation
        # all_predictions.append(predictions.cpu())
        # all_targets.append(targets.cpu())
        
        all_predictions.append(predictions)
        all_targets.append(targets)
        
        # Update tqdm progress bar if the loader is wrapped with tqdm
        if hasattr(loader, 'set_postfix'):
            loader.set_postfix(
                loss=f"{loss_val:.4f}",
                acc=f"{correct / seen * 100:.2f}%",
            )


    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    metrics = calculate_metrics(
        predictions,
        targets,
        num_classes=num_classes,
    )

    avg_loss = total_loss / max(len(loader), 1)

    return avg_loss, metrics
