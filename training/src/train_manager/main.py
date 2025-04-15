import argparse

from .simulator_runner import SimulatorRunner
from .train_handler import TrainHandler
from .train_manager import TrainManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Manager CLI")
    parser.add_argument("--base_folder", type=str, default="training_iterations", help="Base folder for training iterations")
    parser.add_argument("--iterations", type=int, default=5, help="Number of training iterations")
    parser.add_argument("--games", type=int, default=200, help="Number of games per iteration")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")

    args = parser.parse_args()

    simulator_runner = SimulatorRunner(cargo_manifest_path="r2048_core/Cargo.toml")
    train_handler = TrainHandler(
        num_epochs=args.num_epochs,
        batch_size=2048,
        learning_rate=0.001,
        weight_decay=0.0001,
        train_val_split=0.8,
        seed=42
    )

    train_manager = TrainManager(
        simulator_runner=simulator_runner,
        train_handler=train_handler,
        base_folder=args.base_folder,
        iterations=args.iterations,
        games=args.games,
        num_epochs=args.num_epochs
    )

    train_manager.run_training_cycle()