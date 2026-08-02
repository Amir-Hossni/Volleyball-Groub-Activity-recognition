import torch
from pathlib import Path
from torchmetrics.classification import MulticlassF1Score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix


# Cache MulticlassF1Score instances so they are not recreated every epoch.
# Keyed by (num_classes, f1_average, device) to keep the metric state isolated
# per configuration while avoiding the per-call construction overhead.
_f1_metric_cache = {}


def _get_f1_metric(num_classes, f1_average, device):
    key = (num_classes, f1_average, str(device))
    metric = _f1_metric_cache.get(key)
    if metric is None:
        metric = MulticlassF1Score(
            num_classes=num_classes,
            average=f1_average,
        ).to(device)
        _f1_metric_cache[key] = metric
    return metric


def calculate_metrics(
    predictions,
    targets, num_classes=None, class_names=None, f1_average="macro"):
    """
    Calculate classification metrics

    Args:
        predictions:
            predicted class indices tensor

        targets:
            ground truth labels tensor

        num_classes:
            number of classes

        class_names:
            list of class names for classification report

        f1_average:
            macro / weighted / micro

    Returns:
        metrics dictionary
    """


    # Accuracy

    correct = ( predictions == targets).sum().item()
    total = targets.numel()
    accuracy = (100 * correct / total)

    # F1 Score using torchmetrics

    if num_classes is None:

        num_classes = (torch.max(targets).item() + 1 )


    f1_metric = _get_f1_metric(num_classes, f1_average, targets.device)

    f1 = f1_metric(
        predictions,
        targets
    ).item()

    # MulticlassF1Score is stateful (accumulates TP/FP/FN counters).
    # Reset after each call so the cached metric only reflects the
    # current epoch's data, matching the original per-call behavior.
    f1_metric.reset()



    metrics = {
        "accuracy": accuracy,
        "f1_score": f1
    }



    # Classification report
    # Used only for final test

    if class_names is not None:

        preds_cpu = predictions.cpu().numpy()
        targets_cpu = targets.cpu().numpy()

        metrics["classification_report"] = classification_report(
            targets_cpu,
            preds_cpu,
            target_names=class_names,
            digits=2
        )

        metrics["confusion_matrix"] = confusion_matrix(
            targets_cpu,
            preds_cpu
        )


    return metrics


def _to_numpy(data):
    """Move tensors to CPU and convert to NumPy, passing other types through."""
    if hasattr(data, "cpu") and hasattr(data, "numpy"):
        return data.cpu().numpy()
    return data


def save_confusion_matrix(
    predictions,
    targets,
    class_names=None,
    save_path=None,
    normalize=None,
    title="Confusion Matrix",
):
    """
    Build and save a confusion matrix as a PNG image file.

    Args:
        predictions:
            predicted class indices (tensor or numpy array)

        targets:
            ground truth class indices (tensor or numpy array)

        class_names (list, optional):
            display names for each class index. When omitted, numeric
            labels 0..max_label are used.

        save_path (str or Path, optional):
            where to save the PNG file. If None, the figure is returned
            instead (e.g. for TensorBoard logging).

        normalize (str, optional):
            None, "true", "pred" or "all" -- passed to sklearn's
            ConfusionMatrixDisplay.

        title (str):
            figure title

    Returns:
        matplotlib.figure.Figure if save_path is None, else None.
    """
    

    matplotlib.use("Agg")

    

    preds_cpu = _to_numpy(predictions)
    targets_cpu = _to_numpy(targets)

    # If class names are not provided, infer the label range from the data.
    if class_names is None:
        max_label = int(
            max(
                int(targets_cpu.max()) if targets_cpu.size > 0 else 0,
                int(preds_cpu.max()) if preds_cpu.size > 0 else 0,
            )
        )
        class_names = [str(i) for i in range(max_label + 1)]

    labels = list(range(len(class_names)))

    display = ConfusionMatrixDisplay.from_predictions(
        targets_cpu,
        preds_cpu,
        labels=labels,
        display_labels=class_names,
        normalize=normalize,
        cmap="Blues",
        colorbar=True,
    )
    display.ax_.set_title(title)
    display.figure_.set_size_inches(
        max(6.0, len(class_names) * 0.9),
        max(6.0, len(class_names) * 0.75),
    )
    display.figure_.tight_layout()

    if save_path is None:
        return display.figure_

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    display.figure_.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(display.figure_)
    return None
