import torch

from utlis.tensorboard import create_writer, log_metrics
from utlis.metrics import accuracy, create_f1_metric

from utlis.checkpoint import save_checkpoint

from .eval_b2 import evaluate

# import sys
# from pathlib import Path

# ROOT_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(ROOT_DIR))



def train_one_epoch(model,train_loader,criterion,optimizer,device):

    model.train()

    total_loss = 0
    total_acc = 0


    for batch in train_loader:

        images = batch["images"].to(device)

        labels = batch["scene_label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs,dim=1)

        total_acc += accuracy(predictions,labels)

    epoch_loss = total_loss / len(train_loader)

    epoch_acc = total_acc / len(train_loader)


    return epoch_loss, epoch_acc




def train( model, train_loader, val_loader, criterion,
          optimizer, device, epochs, save_path, num_classes):


    writer = create_writer("runs/B2_training")


    f1_metric = create_f1_metric(num_classes, device)

    best_f1 = 0

    for epoch in range(epochs):

        print("=" * 50)
        print(f"Starting Epoch [{epoch+1}/{epochs}]")
        print("=" * 50)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )


        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            f1_metric
        )


        # TensorBoard logging

        log_metrics(
            writer,
            {
                "Loss/train": train_loss,
                "Loss/val": val_loss,

                "Accuracy/train": train_acc,
                "Accuracy/val": val_acc,

                "F1/val": val_f1,

                "Learning_rate":
                    optimizer.param_groups[0]["lr"]
            },
            epoch
        )


        print(
            f"""
Epoch [{epoch+1}/{epochs}]

Train Loss : {train_loss:.4f}
Train Acc  : {train_acc:.2f}%

Val Loss   : {val_loss:.4f}
Val Acc    : {val_acc:.2f}%
Val F1     : {val_f1:.4f}

"""
        )



        # Save best checkpoint

        if val_f1 > best_f1:


            best_f1 = val_f1


            save_checkpoint(
                save_path,
                model.module if isinstance(model, torch.nn.DataParallel)
                else model,
                optimizer,
                epoch + 1,
                {
                    "val_f1": val_f1,
                    "val_acc": val_acc
                }
            )


            print(
                "Best model saved!"
            )



    writer.close()