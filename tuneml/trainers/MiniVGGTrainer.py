import os
import time
from typing import Optional

import mlflow
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from tuneml.models.vgg.MiniVGG import MiniVGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixup(x, y, alpha=0.2):
    """
    Applies Mixup to a batch of inputs and labels.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes loss for Mixup-augmented batches.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MiniVGGTrainer:
    """
    Trainer class for the MiniVGG model.
    """

    def __init__(
        self,
        train_dataset,
        test_dataset,
        batch_size: int = 32,
        lr: float = 1e-3,
        max_lr: float = 1e-2,
        ckpt_path: str = "weights/best_mini_vgg.pth",
        load_from_checkpoint: bool = False,
        mixup_alpha: float = 0.2,
        mixup_probability: float = 0.3,
        label_smoothing: float = 0.1,
        mlflow_enabled: bool = False,
        mlflow_log_model: bool = False,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_lr = max_lr
        self.ckpt_path = ckpt_path
        self.mixup_alpha = mixup_alpha
        self.mixup_probability = mixup_probability
        self.mlflow_enabled = mlflow_enabled
        self.mlflow_log_model = mlflow_log_model

        if len(self.train_dataset) == 0:
            raise ValueError("train_dataset is empty")
        if len(self.test_dataset) == 0:
            raise ValueError("test_dataset is empty")

        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_dl = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        num_classes = len(self.train_dataset.classes)
        self.model = MiniVGG(num_classes=num_classes).to(device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Scheduler depends on epoch count, so it is created in fit().
        self.scheduler: Optional[OneCycleLR] = None

        self.best_accuracy = 0.0
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []

        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            self.load()

    def _train_step(self, spectrograms: torch.Tensor, labels: torch.Tensor) -> float:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        if np.random.rand() < self.mixup_probability:
            spectrograms, labels_a, labels_b, lam = mixup(spectrograms, labels, alpha=self.mixup_alpha)
            output = self.model(spectrograms)
            loss = mixup_criterion(self.criterion, output, labels_a, labels_b, lam)
        else:
            output = self.model(spectrograms)
            loss = self.criterion(output, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return float(loss.item())

    def _eval_epoch(self):
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for spectrograms, labels in self.test_dl:
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)

                output = self.model(spectrograms)
                loss = self.criterion(output, labels)
                test_loss += loss.item() * labels.size(0)

                predicted = torch.argmax(output, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / max(1, len(self.test_dataset))
        accuracy = 100.0 * correct / max(1, total)
        return avg_test_loss, accuracy

    def save(self, ckpt_path=None):
        """
        Saves a checkpoint at ckpt_path.
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt_dir = os.path.dirname(self.ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_accuracy": self.best_accuracy,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "accuracies": self.accuracies,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "max_lr": self.max_lr,
            "class_to_idx": getattr(self.train_dataset, "class_to_idx", None),
        }
        torch.save(ckpt, self.ckpt_path)

    def load(self, ckpt_path=None):
        """
        Loads a checkpoint from ckpt_path.
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = torch.load(self.ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.best_accuracy = float(ckpt.get("best_accuracy", 0.0))
        self.train_losses = ckpt.get("train_losses", [])
        self.test_losses = ckpt.get("test_losses", [])
        self.accuracies = ckpt.get("accuracies", [])

        scheduler_state = ckpt.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

    def fit(self, epochs: int):
        """
        Runs model training and evaluation for a given number of epochs.
        """
        if len(self.train_dl) == 0:
            raise RuntimeError("No training batches available. Check train_dataset and batch_size.")

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            steps_per_epoch=len(self.train_dl),
            epochs=epochs,
            pct_start=0.1,
        )

        print("Beginning MiniVGG training...")
        print(time.strftime("%Y-%m-%d %H:%M"))

        for epoch in range(epochs):
            self.model.train()
            running_train_loss = 0.0

            for spectrograms, labels in tqdm(
                self.train_dl,
                desc=f"Epoch {epoch + 1}/{epochs} - Training",
            ):
                loss_value = self._train_step(spectrograms, labels)
                running_train_loss += loss_value * spectrograms.size(0)

            avg_train_loss = running_train_loss / max(1, len(self.train_dataset))
            avg_test_loss, accuracy = self._eval_epoch()

            self.train_losses.append(avg_train_loss)
            self.test_losses.append(avg_test_loss)
            self.accuracies.append(accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}, "
                f"Accuracy: {accuracy:.2f}%"
            )

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save(self.ckpt_path)
                print(f"New best checkpoint saved at {self.ckpt_path} with accuracy {self.best_accuracy:.2f}%")

                if self.mlflow_enabled and self.mlflow_log_model:
                    mlflow.pytorch.log_model(self.model, name="best_mini_vgg")

            if self.mlflow_enabled:
                mlflow.log_metrics(
                    {
                        "train_loss": avg_train_loss,
                        "test_loss": avg_test_loss,
                        "accuracy": accuracy,
                    },
                    step=epoch + 1,
                )

        print(f"Training completed. Best test accuracy: {self.best_accuracy:.2f}%")
        return self.train_losses, self.test_losses, self.accuracies