from pathlib import Path
from typing import Optional, Dict, Any, List

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.base import BaseModel
from .data.dataset import GameDataset
from .data.batch_sampler import TrajectoryBatchSampler, collate_trajectories
from .types import TrainResult
from .utils.config import TrainingConfig
from .utils.metrics import compute_metrics


class PolicyGradientTrainer:
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
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_length = 0
        num_trajectories = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # Zero gradients at the start of each batch
            self.optimizer.zero_grad()
            
            # Process each trajectory in the batch
            batch_loss = 0.0
            for trajectory in batch:
                states = trajectory.states.to(self.device)
                actions = trajectory.actions.to(self.device)
                length = trajectory.length
                
                # Add channel dimension to states
                states = states.unsqueeze(1)  # Shape: [batch_size, 1, 4, 4]
                
                # Get action probabilities
                action_probs = self.model(states)
                
                # Calculate policy loss (negative log probability of taken actions)
                log_probs = torch.log(action_probs + 1e-10)
                selected_log_probs = log_probs[range(length), actions]
                
                # Calculate policy gradient loss using trajectory length as reward
                loss = -torch.mean(selected_log_probs * length)
                batch_loss += loss
                
                total_length += length
                num_trajectories += 1
            
            # Average the loss over trajectories in the batch
            batch_loss = batch_loss / len(batch)
            
            # Backward pass for the entire batch
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": total_loss / num_trajectories,
                "avg_length": total_length / num_trajectories,
            })
        
        metrics = {
            "train_loss": total_loss / num_trajectories,
            "train_avg_length": total_length / num_trajectories,
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_length = 0
        num_trajectories = 0
        
        with torch.no_grad():
            for batch in val_loader:
                for trajectory in batch:
                    states = trajectory.states.to(self.device)
                    actions = trajectory.actions.to(self.device)
                    length = trajectory.length
                    
                    # Add channel dimension to states
                    states = states.unsqueeze(1)  # Shape: [batch_size, 1, 4, 4]
                    
                    # Get action probabilities
                    action_probs = self.model(states)
                    
                    # Calculate policy loss
                    log_probs = torch.log(action_probs + 1e-10)
                    selected_log_probs = log_probs[range(length), actions]
                    loss = -torch.mean(selected_log_probs * length)
                    
                    total_loss += loss.item()
                    total_length += length
                    num_trajectories += 1
        
        metrics = {
            "val_loss": total_loss / num_trajectories,
            "val_avg_length": total_length / num_trajectories,
        }
        
        return metrics
    
    def train(
        self,
        train_dataset: GameDataset,
        val_dataset: GameDataset,
    ) -> TrainResult:
        # Calculate max samples per batch based on batch_size
        max_samples_per_batch = self.config.batch_size * 4 * 4  # Assuming 4x4 grid
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=TrajectoryBatchSampler(train_dataset, max_samples_per_batch),
            num_workers=self.config.num_workers,
            collate_fn=collate_trajectories,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_trajectories,
        )
        
        result = TrainResult(
            train_loss=[],
            train_avg_length=[],
            val_loss=[],
            val_avg_length=[],
        )
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Save best model
            if val_metrics["val_avg_length"] > result.best_val_length:
                self.save_model(self.config.model_save_path)
            
            # Update result
            result.train_loss.append(train_metrics["train_loss"])
            result.train_avg_length.append(train_metrics["train_avg_length"])
            result.val_loss.append(val_metrics["val_loss"])
            result.val_avg_length.append(val_metrics["val_avg_length"])
        
        return result
    
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