from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .models.base import BaseModel
from .data.dataset import GameDataset
from .utils.config import TrainingConfig
from .utils.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project="2048-cnn",
                config=config.to_dict(),
            )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target.argmax(dim=1)).sum().item()
            total_samples += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": total_loss / (batch_idx + 1),
                "acc": total_correct / total_samples,
            })
        
        metrics = {
            "train_loss": total_loss / len(train_loader),
            "train_acc": total_correct / total_samples,
        }
        
        if self.config.use_wandb:
            wandb.log(metrics)
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target.argmax(dim=1)).sum().item()
                total_samples += target.size(0)
        
        metrics = {
            "val_loss": total_loss / len(val_loader),
            "val_acc": total_correct / total_samples,
        }
        
        if self.config.use_wandb:
            wandb.log(metrics)
        
        return metrics
    
    def train(
        self,
        train_dataset: GameDataset,
        val_dataset: GameDataset,
    ) -> Dict[str, Any]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        
        best_val_loss = float("inf")
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics["val_loss"])
            
            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_model(self.config.model_save_path)
            
            # Update history
            for k, v in train_metrics.items():
                history[k].append(v)
            for k, v in val_metrics.items():
                history[k].append(v)
        
        return history
    
    def save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )
    
    def load_model(self, path: Path) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.config = TrainingConfig.from_dict(checkpoint["config"]) 