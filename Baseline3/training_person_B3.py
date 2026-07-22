import torch

from utlis.tensorboard import create_writer, log_metrics
from utlis.metrics import calculate_metrics
from utlis.checkpoint import save_checkpoint
from utlis.early_stopping import EarlyStopping

from .eval_person_B3 import evaluate_person



def train_person_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0

    all_predictions = []
    all_targets = []


    for batch in train_loader:


        images = batch["image"].to(device)

        labels = batch["player_label"].to(device)



        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()



        total_loss += loss.item()



        predictions = torch.argmax(
            outputs,
            dim=1
        )



        all_predictions.append(
            predictions
        )


        all_targets.append(
            labels
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
        num_classes=9
    )



    epoch_loss = (
        total_loss /
        len(train_loader)
    )



    return epoch_loss, metrics







def train_person_B3(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path
):


    writer = create_writer(
        "runs/B3_person"
    )



    early_stopping = EarlyStopping(
        patience=10,
        mode="max"
    )



    best_f1 = 0



    for epoch in range(epochs):


        print("="*50)

        print(
            f"Epoch [{epoch+1}/{epochs}]"
        )

        print("="*50)




        train_loss, train_metrics = train_person_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )




        val_loss, val_metrics = evaluate_person(
            model,
            val_loader,
            criterion,
            device
        )





        log_metrics(
            writer,
            {

                "Loss/train":
                    train_loss,


                "Accuracy/train":
                    train_metrics["accuracy"],


                "F1/train":
                    train_metrics["f1_score"],



                "Loss/val":
                    val_loss,


                "Accuracy/val":
                    val_metrics["accuracy"],


                "F1/val":
                    val_metrics["f1_score"]

            },

            epoch
        )





        print(
            f"""
Train Loss : {train_loss:.4f}
Train Acc  : {train_metrics["accuracy"]:.2f}%
Train F1   : {train_metrics["f1_score"]:.4f}

Val Loss   : {val_loss:.4f}
Val Acc    : {val_metrics["accuracy"]:.2f}%
Val F1     : {val_metrics["f1_score"]:.4f}

"""
        )





        if val_metrics["f1_score"] > best_f1:


            best_f1 = val_metrics["f1_score"]



            save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch,
                {
                    "val_f1":
                        best_f1,

                    "val_acc":
                        val_metrics["accuracy"]
                }
            )



            print(
                "Best model saved"
            )





        if early_stopping(
            val_metrics["f1_score"]
        ):

            print(
                "Early stopping triggered"
            )

            break




    writer.close()