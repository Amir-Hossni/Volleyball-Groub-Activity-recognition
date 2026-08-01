for epoch in range(EPOCHS):
    total_acc_train = total_loss_train = total_loss_val = total_acc_val = 0
    seen_train = seen_val = 0

    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False)
    # for inputs, labels in train_bar:
    #     inputs = inputs.to(device, non_blocking=True)
    #     labels = labels.to(device, non_blocking=True)
    for batch in train_bar:

        inputs = batch["images"].to(
            device,
            non_blocking=True
        )

        labels = batch["scene_label"].to(
            device,
            non_blocking=True
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        total_loss_train += train_loss.item()
        train_loss.backward()
        optimizer.step()

        batch_correct = (torch.argmax(outputs, 1) == labels).sum().item()
        total_acc_train += batch_correct
        seen_train += labels.size(0)

        train_bar.set_postfix(
            loss=f"{train_loss.item():.4f}",
            acc=f"{total_acc_train / seen_train * 100:.2f}%",
        )
        
    true_labels = []
    predicted_labels = []
    model.eval()
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False)
    with torch.no_grad():
        # for inputs, labels in val_bar:
        #     inputs = inputs.to(device, non_blocking=True)
        #     labels = labels.to(device, non_blocking=True)
        for batch in val_bar:

            inputs = batch["images"].to(
            device,
            non_blocking=True
            )

            labels = batch["scene_label"].to(
                device,
                non_blocking=True
            )
            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

            val_loss = criterion(outputs, labels)
            total_loss_val += val_loss.item()

            batch_correct = (torch.argmax(outputs, 1) == labels).sum().item()
            total_acc_val += batch_correct
            seen_val += labels.size(0)

            val_bar.set_postfix(
                loss=f"{val_loss.item():.4f}",
                acc=f"{total_acc_val / seen_val * 100:.2f}%",
            )

    avg_train_loss = total_loss_train / len(train_loader)
    avg_val_loss = total_loss_val / len(val_loader)
    scheduler.step()

    # Save checkpoint every epoch
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }
    
    torch.save(
        checkpoint,
        f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
    )
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
    
        torch.save(
            checkpoint,
            f"{checkpoint_dir}/best_model.pth"
        )
    
        print(f"The Best model saved (Val Loss = {avg_val_loss:.4f})")
    

    total_loss_train_plot.append(round(avg_train_loss, 4))
    total_loss_validation_plot.append(round(avg_val_loss, 4))
    total_acc_train_plot.append(round(total_acc_train / seen_train * 100, 4))
    total_acc_validation_plot.append(round(total_acc_val / seen_val * 100, 4))
    val_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro"
    )
    total_f1_validation_plot.append(round(val_f1, 4))
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Train Accuracy: {total_acc_train / seen_train * 100:.2f}% "
        f"Validation Loss: {avg_val_loss:.4f} "
        f"Validation Accuracy: {total_acc_val / seen_val * 100:.2f}%"
        f"Validation F1: {val_f1:.4f}"
    )
    print("=" * 25) 