import torch
from torch.utils.tensorboard import SummaryWriter

from .eval_b2 import evaluate



def train_one_epoch(model,train_loader, criterion, optimizer, device):

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for batch in train_loader:

        images = batch["images"].to(device)
        labels = batch["scene_label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs,dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total


    return epoch_loss, epoch_acc




def train(model,train_loader,val_loader,criterion,optimizer,device,epochs,save_path):

    writer = SummaryWriter("runs/B2_training")

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
            device
        )


        # TensorBoard

        writer.add_scalar(
            "Loss/train",
            train_loss,
            epoch
        )

        writer.add_scalar(
            "Loss/val",
            val_loss,
            epoch
        )


        writer.add_scalar(
            "Accuracy/train",
            train_acc,
            epoch
        )


        writer.add_scalar(
            "Accuracy/val",
            val_acc,
            epoch
        )


        writer.add_scalar(
            "F1/val",
            val_f1,
            epoch
        )


        writer.add_scalar(
            "Learning_rate",
            optimizer.param_groups[0]["lr"],
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



        # Save best model according to F1 score

        if val_f1 > best_f1:


            best_f1 = val_f1


            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": (
                        model.module.state_dict()
                        if isinstance(model, torch.nn.DataParallel)
                        else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_acc": val_acc
                },
                save_path
            )


            print(
                "Best model saved!"
            )



    writer.close()