from pathlib import Path
from typing import Optional, Dict, Any, List

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.model_loader import ModelLoader
from ..models.base import BaseModel
from ..data.dataset import GameDataset
from ..data.batch_sampler import TrajectoryBatchSampler, collate_trajectories
from ..types import TrainResult, Trajectory, ValidationMetrics
from ..utils.config import TrainingConfig
from .validator import Validator, DefaultValidator


logger = logging.getLogger(__name__)

class PolicyGradientTrainer:
    def __init__(
        self,
        model_loader: ModelLoader,
        config: TrainingConfig,
        device: Optional[str] = None,
        validator: Validator = None,
    ):
        self.model_loader = model_loader
        self.model = model_loader.load_model()
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.model.train()
        self.validator = validator or DefaultValidator()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        total_loss = 0
        num_trajectories = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        batch: List[Trajectory]
        for batch in progress_bar:
            # Zero gradients at the start of each batch
            self.optimizer.zero_grad()
            
            # Process each trajectory in the batch
            batch_loss = 0.0
            trajectory: Trajectory
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
                loss = -torch.sum(selected_log_probs) * length
                batch_loss += loss
                
                num_trajectories += 1
            
            # Average the loss over trajectories in the entire dataset
            batch_loss = batch_loss / len(train_loader.dataset)
            
            # Backward pass for the entire batch
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": total_loss / num_trajectories,
            })
        
        metrics = {
            "train_loss": total_loss / num_trajectories,
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> ValidationMetrics:
        return self.validator.validate(self.model, val_loader, self.device)
        
    
    def train(
        self,
        train_dataset: GameDataset,
        val_dataset: GameDataset,
    ) -> TrainResult:
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=TrajectoryBatchSampler(train_dataset, self.config.batch_size),
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

        best_val_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics: ValidationMetrics = self.validate(val_loader)
            
            # Save best model
            if val_metrics.val_loss < best_val_loss:
                best_val_loss = val_metrics.val_loss
                logger.info(f"New best validation loss: {best_val_loss}")
                self.model_loader.save_model()
            
            # Update result
            result.train_loss.append(train_metrics["train_loss"])
            result.val_loss.append(val_metrics.val_loss)
            result.val_exp_length.append(val_metrics.exp_length)
        
        return result
