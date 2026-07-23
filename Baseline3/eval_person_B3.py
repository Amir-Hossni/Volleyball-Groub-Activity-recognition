import torch
from utlis.metrics import calculate_metrics


@torch.no_grad()
def evaluate_person(
        model,
        loader,
        criterion,
        device,
        num_classes=9
):

    model.eval()

    total_loss = 0

    all_predictions = []
    all_targets = []


    for batch in loader:


        images = batch["images"].to(device)

        labels = batch["player_label"].to(device)

        B, P, C, H, W = images.shape
        
        images = images.view( B * P, C, H, W)
        
        labels = labels.view(-1)

        outputs = model(images)


        loss = criterion(outputs, labels)


        total_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # remove padded players
        mask = labels != -1
        
        predictions = predictions[mask]

        labels = labels[mask]
        
        all_predictions.append(
            predictions
        )

        all_targets.append(labels)

    predictions = torch.cat(all_predictions)

    targets = torch.cat(all_targets)


    metrics = calculate_metrics(
        predictions,
        targets,
        num_classes=num_classes,
        f1_average="weighted"
    )


    avg_loss = (
        total_loss /
        len(loader)
    )


    return avg_loss, metrics