import logging
from pathlib import Path

from model_training.main import PolicyGradientTrainer, ModelLoader, setup_environment
from model_training.preprocess.data_loader import GameDataLoader
from model_training.utils.config import TrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainHandler:
    def __init__(self, num_epochs: int = 100, batch_size: int = 2048, learning_rate: float = 0.001, weight_decay: float = 0.0001, train_val_split: float = 0.8, seed: int = 42):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_val_split = train_val_split
        self.seed = seed

        # Set up environment
        setup_environment(self.seed)

    def train_model(self, log_path: Path, save_dir: Path, npz_save_path: Path) -> None:
        # Load trajectories
        data_loader = GameDataLoader([log_path])
        train_trajectories, val_trajectories = data_loader.get_datasets()

        # Create model and trainer
        config = TrainingConfig(
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            train_val_split=self.train_val_split,
            model_save_path=save_dir / "best_model.pt",
            npz_save_path=npz_save_path,
        )
        model_loader = ModelLoader(config)
        trainer = PolicyGradientTrainer(model_loader, config)

        # Train
        logger.info("Starting training...")
        result = trainer.train(train_trajectories, val_trajectories)

        logger.info(f"Training completed!")
        logger.info(f"Best validation length: {result.best_val_length} at epoch {result.best_epoch}")

        # Load the best model and save it as npz
        model_loader.load_model()
        model_loader.save_model_as_npz(npz_save_path)
        logger.info(f"Model saved as npz to {npz_save_path}")