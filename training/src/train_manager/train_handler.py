import subprocess
import logging
from pathlib import Path
from training.model_training.src.train import PolicyGradientTrainer, ModelLoader, create_config, setup_environment
from training.model_training.src.preprocess.data_loader import GameDataLoader
from training.model_training.src.utils.config import TrainingConfig

class TrainHandler:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


    def train_model(log_path: Path, save_dir: Path, npz_save_path: Path, num_epochs: int = 100, batch_size: int = 2048, learning_rate: float = 0.001, weight_decay: float = 0.0001, train_val_split: float = 0.8, seed: int = 42):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Set up environment
        setup_environment(seed)

        # Load trajectories
        data_loader = GameDataLoader(log_path)
        train_trajectories, val_trajectories = data_loader.get_datasets()

        # Create model and trainer
        config = TrainingConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_val_split=train_val_split,
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