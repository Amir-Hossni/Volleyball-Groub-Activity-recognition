import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def _to_numpy(data):
    """Move tensors to CPU and convert to NumPy, passing other types through.

    Duck-typed check for tensor-like objects (e.g. torch.Tensor) so the same
    inputs work as with metrics.py, without importing PyTorch.
    """
    if hasattr(data, "cpu") and hasattr(data, "numpy"):
        return data.cpu().numpy()
    return data


def _torchmetrics_macro_f1(targets, preds, num_classes):
    """Replicate torchmetrics MulticlassF1Score macro averaging exactly.

    torchmetrics computes per-class F1 over all ``num_classes`` labels
    (classes with no true positives get F1 = 0 via zero_division) and then
    averages only over classes where ``tp + fp + fn > 0`` -- i.e. classes
    absent from both the target and the predictions are excluded from the
    macro denominator. This mirrors the ``_adjust_weights_safe_divide``
    weight masking we would otherwise lose by using sklearn's plain macro
    mean over all ``num_classes``.
    """
    labels = list(range(num_classes))

    per_class_f1 = f1_score(
        targets,
        preds,
        labels=labels,
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(targets, preds, labels=labels)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    diag = np.diag(cm)
    # tp + fp + fn per class
    present = (row_sum + col_sum - diag) > 0

    if not present.any():
        return 0.0
    return float(per_class_f1[present].mean())


def calculate_metrics(
    predictions,
    targets,
    num_classes=None,
    class_names=None,
    f1_average="macro",
):
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

    # Move to CPU and convert to NumPy if tensors,
    # otherwise pass through as-is (e.g. NumPy arrays).
    preds_cpu = _to_numpy(predictions)
    targets_cpu = _to_numpy(targets)

    # Accuracy (sklearn)
    accuracy = 100.0 * accuracy_score(
        targets_cpu,
        preds_cpu
    )

    # F1 Score (sklearn).
    # labels=list(range(num_classes)) mirrors MulticlassF1Score behavior:
    # every class is included in the average, and classes with no samples
    # contribute F1 = 0 via zero_division, exactly matching torchmetrics.
    if num_classes is None:

        num_classes = int(targets_cpu.max()) + 1

    if f1_average == "macro":
        # torchmetrics excludes classes absent from both targets and preds
        # from the macro denominator; plain sklearn macro does not.
        f1 = _torchmetrics_macro_f1(targets_cpu, preds_cpu, num_classes)
    else:
        f1 = f1_score(
            targets_cpu,
            preds_cpu,
            average=f1_average,
            labels=list(range(num_classes)),
            zero_division=0,
        )

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1
    }

    # Classification report
    # Used only for final test

    if class_names is not None:

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