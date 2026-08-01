import torch
from torchmetrics.classification import MulticlassF1Score
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
    total = targets.size(0)
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