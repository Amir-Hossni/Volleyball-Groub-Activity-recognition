import torch
from torchmetrics.classification import MulticlassF1Score



def accuracy(predictions, targets):
    """
    Calculate classification accuracy

    Args:
        predictions:
            predicted class indices

        targets:
            ground truth labels

    Returns:
        accuracy percentage
    """

    correct = (
        predictions == targets
    ).sum().item()

    total = targets.size(0)

    return 100 * correct / total




def create_f1_metric(num_classes, device):
    """
    Create F1 score metric

    Args:
        num_classes:
            number of classes

        device:
            cuda or cpu

    Returns:
        torchmetrics F1 metric
    """

    metric = MulticlassF1Score(
        num_classes=num_classes,
        average="macro"
    )

    return metric.to(device)




def f1_score_metric(metric, predictions, targets):
    """
    Calculate F1 score using torchmetrics

    Args:
        metric:
            torchmetrics F1 metric object

        predictions:
            predicted class indices

        targets:
            ground truth labels

    Returns:
        F1 score
    """

    score = metric(
        predictions,
        targets
    )

    return score.item()