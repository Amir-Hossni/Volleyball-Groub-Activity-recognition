import torch

from utlis.metrics import calculate_metrics



@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    adapter,
    num_classes
):


    model.eval()


    total_loss = 0


    all_predictions = []
    all_targets = []



    for batch in loader:


        inputs, targets = adapter(batch)


        inputs = inputs.to(device)

        targets = targets.to(device)



        outputs = model(inputs)



        loss = criterion(
            outputs,
            targets
        )


        total_loss += loss.item()



        predictions = torch.argmax(
            outputs,
            dim=1
        )



        all_predictions.append(
            predictions
        )

        all_targets.append(
            targets
        )



    predictions = torch.cat(
        all_predictions
    )


    targets = torch.cat(
        all_targets
    )



    metrics = calculate_metrics(
        predictions,
        targets,
        num_classes=num_classes
    )



    avg_loss = (
        total_loss /
        len(loader)
    )


    return avg_loss, metrics
