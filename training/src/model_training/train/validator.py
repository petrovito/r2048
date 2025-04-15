from abc import ABC, abstractmethod
from typing import List
import torch
from ..types import Trajectory, ValidationMetrics

class Validator(ABC):
    @abstractmethod
    def validate(self, model, val_loader, device) -> ValidationMetrics:
        """Abstract method to validate a model.

        Args:
            model: The model to validate.
            val_loader: DataLoader for validation data.
            device: Device to run the validation on.

        Returns:
            ValidationMetrics containing validation loss and other metrics.
        """
        pass

class DefaultValidator(Validator):
    def validate(self, model, val_loader, device) -> ValidationMetrics:
        model.eval()
        total_loss = 0.0
        total_length = 0
        num_trajectories = 0

        with torch.no_grad():
            for batch in val_loader:
                trajectory: Trajectory
                for trajectory in batch:
                    states = trajectory.states.to(device)
                    actions = trajectory.actions.to(device)
                    length = trajectory.length

                    # Add channel dimension to states
                    states = states.unsqueeze(1)  # Shape: [batch_size, 1, 4, 4]

                    # Get action probabilities
                    action_probs = model(states)

                    # Calculate policy loss
                    log_probs = torch.log(action_probs + 1e-10)
                    selected_log_probs = log_probs[range(length), actions]
                    loss = -torch.sum(selected_log_probs * length)

                    total_loss += loss.item()
                    total_length += length
                    num_trajectories += 1

        return ValidationMetrics(
            val_loss=total_loss / num_trajectories,
            exp_length=total_length / num_trajectories,
        )