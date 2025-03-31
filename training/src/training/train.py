import argparse
import logging
from pathlib import Path
import random
import torch
import numpy as np

from .utils.config import TrainingConfig
from .utils.data_loader import GameDataLoader
from .utils.model_loader import ModelLoader

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
        default=32,
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
    
    # Model arguments
    parser.add_argument(
        "--input_channels",
        type=int,
        default=1,
        help="Number of input channels",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate",
    )
    
    # Logging arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
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
        input_channels=args.input_channels,
        use_wandb=args.use_wandb,
        model_save_path=args.save_dir / "best_model.pt",
    )


def main():
    args = parse_args()
    
    # Set up environment
    setup_environment(args.seed)
    
    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = create_config(args)
    
    # Initialize components
    data_loader = GameDataLoader(
        log_path=args.log_path,
        train_val_split=config.train_val_split,
        seed=args.seed
    )
    
    model_loader = ModelLoader(
        config=config,
        load_model_path=args.load_model
    )
    
    # Get datasets and trainer
    train_dataset, val_dataset = data_loader.get_datasets()
    trainer = model_loader.get_trainer()
    
    # Train the model
    history = trainer.train(train_dataset, val_dataset)
    
    # Log final results
    logger.info("Training completed!")
    logger.info(f"Best validation reward: {max(history['val_reward'])}")
    logger.info(f"Final training reward: {history['train_reward'][-1]}")
    logger.info(f"Final validation reward: {history['val_reward'][-1]}")


if __name__ == "__main__":
    main() 