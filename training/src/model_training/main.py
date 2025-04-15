import argparse
import logging
from pathlib import Path
import random
import torch
import numpy as np

from .models.model_loader import ModelLoader
from .models.policy_network import PolicyNetwork
from .utils.config import TrainingConfig
from .train.trainer import PolicyGradientTrainer
from .preprocess.data_loader import GameDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy network for 2048")
    
    # Data arguments
    parser.add_argument(
        "--log_path",
        type=Path,
        required=True,
        help="Path to the game log file",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("models"),
        help="Directory to save the model and processed data",
    )

    parser.add_argument(
        "--npz_save_path",
        type=Path,
        required=True,
        help="Path to save the model as npz",
    )
    parser.add_argument(
        "--load_model",
        type=Path,
        help="Path to an existing model to continue training from",
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training",
    )

    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def setup_environment(seed: int) -> None:
    """Set up the training environment with random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_config(args) -> TrainingConfig:
    """Create the training configuration."""
    return TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_val_split=args.train_val_split,
        model_save_path=args.save_dir / "best_model.pt",
        npz_save_path=args.npz_save_path,
    )


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    args = parse_args()
    
    # Load trajectories
    data_loader = GameDataLoader(args.log_path)
    train_trajectories, val_trajectories = data_loader.get_datasets()
    
    # Create model and trainer
    config = create_config(args)
    model_loader = ModelLoader(config)
    trainer = PolicyGradientTrainer(model_loader, config)
    
    # Train
    logger.info("Starting training...")
    result = trainer.train(train_trajectories, val_trajectories)
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation length: {result.best_val_length} at epoch {result.best_epoch}")

    # Load the best model and save it as npz
    model_loader.load_model()
    model_loader.save_model_as_npz(args.npz_save_path)
    logger.info(f"Model saved as npz to {args.npz_save_path}")


if __name__ == "__main__":
    main() 