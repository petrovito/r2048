from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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


def collate_trajectories(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length trajectories.
    
    Args:
        batch: List of dictionaries containing trajectory data
        
    Returns:
        Dictionary containing batched trajectory data
    """
    # Get maximum length in this batch
    max_length = max(item["length"].item() for item in batch)
    batch_size = len(batch)
    
    # Initialize tensors for the batch
    states = torch.zeros((batch_size, max_length, 4, 4))
    actions = torch.zeros((batch_size, max_length), dtype=torch.long)
    rewards = torch.zeros((batch_size, max_length))
    returns = torch.zeros((batch_size, max_length))
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill in the tensors
    for i, item in enumerate(batch):
        length = item["length"].item()
        states[i, :length] = item["states"]
        actions[i, :length] = item["actions"]
        rewards[i, :length] = item["rewards"]
        returns[i, :length] = item["returns"]
        lengths[i] = length
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
        "length": lengths,
    }


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
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project="2048-policy-gradient",
                config=config.to_dict(),
            )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_reward = 0
        num_trajectories = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            returns = batch["returns"].to(self.device)
            lengths = batch["length"]
            
            # Process each trajectory in the batch
            for i in range(len(lengths)):
                traj_states = states[i, :lengths[i]]
                traj_actions = actions[i, :lengths[i]]
                traj_returns = returns[i, :lengths[i]]
                
                # Get action probabilities
                action_probs = self.model(traj_states)
                
                # Calculate policy loss (negative log probability of taken actions)
                log_probs = torch.log(action_probs + 1e-10)
                selected_log_probs = log_probs[range(lengths[i]), traj_actions]
                
                # Calculate policy gradient loss
                loss = -torch.mean(selected_log_probs * traj_returns)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_reward += traj_returns[0].item()  # Use first return as trajectory reward
                num_trajectories += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": total_loss / num_trajectories,
                "reward": total_reward / num_trajectories,
            })
        
        metrics = {
            "train_loss": total_loss / num_trajectories,
            "train_reward": total_reward / num_trajectories,
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
        total_reward = 0
        num_trajectories = 0
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                returns = batch["returns"].to(self.device)
                lengths = batch["length"]
                
                # Process each trajectory in the batch
                for i in range(len(lengths)):
                    traj_states = states[i, :lengths[i]]
                    traj_actions = actions[i, :lengths[i]]
                    traj_returns = returns[i, :lengths[i]]
                    
                    # Get action probabilities
                    action_probs = self.model(traj_states)
                    
                    # Calculate policy loss
                    log_probs = torch.log(action_probs + 1e-10)
                    selected_log_probs = log_probs[range(lengths[i]), traj_actions]
                    loss = -torch.mean(selected_log_probs * traj_returns)
                    
                    total_loss += loss.item()
                    total_reward += traj_returns[0].item()
                    num_trajectories += 1
        
        metrics = {
            "val_loss": total_loss / num_trajectories,
            "val_reward": total_reward / num_trajectories,
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
            collate_fn=collate_trajectories,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_trajectories,
        )
        
        best_val_reward = float("-inf")
        history = {
            "train_loss": [],
            "train_reward": [],
            "val_loss": [],
            "val_reward": [],
        }
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Save best model
            if val_metrics["val_reward"] > best_val_reward:
                best_val_reward = val_metrics["val_reward"]
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