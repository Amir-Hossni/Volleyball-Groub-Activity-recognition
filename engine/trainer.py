import torch

from utlis.metrics import calculate_metrics
from utlis.checkpoint import save_checkpoint
from utlis.tensorboard import create_writer, log_metrics
from utlis.early_stopping import EarlyStopping

import time

class Trainer:


    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        adapter,
        num_classes,
        save_path,
        log_name="training",
        epochs=50,
        early_stop_patience=10
    ):


        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

        self.adapter = adapter

        self.num_classes = num_classes

        self.save_path = save_path

        self.epochs = epochs



        self.writer = create_writer(
            f"runs/{log_name}"
        )


        self.early_stopping = EarlyStopping(
            patience=early_stop_patience,
            mode="max"
        )


        self.best_f1 = 0


    def train_one_epoch(self, loader):
        
        
        
        
        
        data_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0
        metric_time = 0.0
        

        self.model.train()

        total_loss = 0

        all_predictions = []

        all_targets = []


        # for batch in loader:
        for batch_idx, batch in enumerate(loader):

            t0 = time.perf_counter()

            inputs, targets = self.adapter(batch)

            
            if batch_idx == 0:
                print("Batch shape:", inputs.shape)
                print("Target shape:", targets.shape)

                print("=" * 50)
                print(f"GPU Memory Allocated : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"GPU Memory Reserved  : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                print("=" * 50)
        

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            ########################
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ########################
            # Forward
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            
            ########################
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            #########################
            
            #  Backward
            loss.backward()
            
            #######################
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            #######################
            
            
            # Optimizer
            self.optimizer.step()
            
            #######################
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            #######################
            
            total_loss += loss.item()



            # Metrics
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.append(predictions)

            all_targets.append(targets)
            t5 = time.perf_counter()
            
            
               # accumulate
            # -------------------------
            data_time += t1 - t0
            forward_time += t2 - t1
            backward_time += t3 - t2
            optimizer_time += t4 - t3
            metric_time += t5 - t4
            ###################################
        predictions = torch.cat(all_predictions)

        targets = torch.cat(all_targets)

        metrics = calculate_metrics(
            predictions,
            targets,
            num_classes=self.num_classes
        )




        print("\n========== TRAIN PROFILE ==========")
        print(f"Data + Transfer : {data_time:.2f} sec")
        print(f"Forward         : {forward_time:.2f} sec")
        print(f"Backward        : {backward_time:.2f} sec")
        print(f"Optimizer       : {optimizer_time:.2f} sec")
        print(f"Metrics         : {metric_time:.2f} sec")
        print("===================================\n")

        return (
            total_loss / len(loader),
            metrics
        )




    def fit(self, train_loader, val_loader):
        from engine.evaluator import evaluate

        for epoch in range(self.epochs):

            print("="*50)
            print(f"Epoch [{epoch+1}/{self.epochs}]")
            print("="*50)

            train_loss, train_metrics = self.train_one_epoch(train_loader)

            val_loss, val_metrics = evaluate(
                self.model,
                val_loader,
                self.criterion,
                self.device,
                self.adapter,
                self.num_classes
            )



            log_metrics(
                self.writer,
                {

                "Loss/train":train_loss,

                "F1/train":
                    train_metrics["f1_score"],

                "Loss/val":val_loss,

                "F1/val":
                    val_metrics["f1_score"]

                },
                epoch
            )



            print(
                f"""
Train Loss: {train_loss:.4f}
Train F1: {train_metrics['f1_score']:.4f}

Val Loss: {val_loss:.4f}
Val F1: {val_metrics['f1_score']:.4f}
"""
            )



            if val_metrics["f1_score"] > self.best_f1:

                self.best_f1 = val_metrics["f1_score"]

                model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                save_checkpoint(
                    self.save_path,
                    model_to_save,
                    self.optimizer,
                    epoch,
                    {
                        "val_f1":
                        self.best_f1
                    }
                )

                print("Best model saved")


            if self.early_stopping(val_metrics["f1_score"]):

                print("Early stopping triggered")

                break


        self.writer.close()