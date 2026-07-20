import torch
from sklearn.metrics import f1_score



@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()


    total_loss = 0

    correct = 0
    total = 0


    all_predictions = []
    all_labels = []



    for batch in loader:


        images = batch["images"].to(device)

        labels = batch["scene_label"].to(device)



        outputs = model(images)



        loss = criterion(
            outputs,
            labels
        )


        total_loss += loss.item()



        predictions = torch.argmax(
            outputs,
            dim=1
        )



        correct += (
            predictions == labels
        ).sum().item()



        total += labels.size(0)



        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_labels.extend(
            labels.cpu().numpy()
        )




    avg_loss = total_loss / len(loader)


    accuracy = (
        100 * correct / total
    )


    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )



    return avg_loss, accuracy, f1