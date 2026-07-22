import torch
from utlis.metrics import calculate_metrics



@torch.no_grad()
def evaluate_group_B3( model, loader, criterion, device, num_classes=8,class_names=None):

    model.eval()

    total_loss = 0

    all_predictions = []
    all_targets = []


    for batch in loader:

        images = batch["images"].to(device)
        labels = batch["scene_label"].to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predictions = torch.argmax(outputs,dim=1)

        all_predictions.append(predictions)

        all_targets.append(labels)


    predictions = torch.cat(all_predictions)

    targets = torch.cat(all_targets)

    metrics = calculate_metrics(
        predictions,
        targets,
        num_classes=num_classes,
        class_names=class_names,
        f1_average="weighted"
    )

    avg_loss = (total_loss / len(loader))


    return avg_loss, metrics