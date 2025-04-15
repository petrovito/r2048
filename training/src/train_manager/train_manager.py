import os
from pathlib import Path

from .simulator_runner import SimulatorRunner
from .train_handler import TrainHandler

class TrainManager:
    def __init__(self, simulator_runner: SimulatorRunner, train_handler: TrainHandler, 
                 base_folder: str = "training_iterations", 
                 iterations: int = 5, games: int = 200, 
                 num_epochs: int = 50):
        self.simulator_runner = simulator_runner
        self.train_handler = train_handler
        self.base_folder = base_folder
        self.iterations = iterations
        self.games = games
        self.num_epochs = num_epochs

    def run_training_cycle(self) -> None:
        base_path = Path(self.base_folder)
        base_path.mkdir(parents=True, exist_ok=True)

        for i in range(self.iterations):
            current_folder = base_path / f"{i:02d}"
            next_folder = base_path / f"{i+1:02d}"

            # Create necessary subfolders
            logs_folder = next_folder / "logs"
            model_folder = next_folder / "model"
            logs_folder.mkdir(parents=True, exist_ok=True)
            model_folder.mkdir(parents=True, exist_ok=True)

            # Determine model paths
            previous_model_npz = current_folder / "model" / "best.npz"
            new_model_npz = model_folder / "best.npz"

            print(f"\nIteration {i+1}/{self.iterations}")
            print(f"Using model from: {previous_model_npz}")

            # Run simulation
            log_file = logs_folder / "game_logs.txt"
            self.simulator_runner.run_simulation(self.games, log_file, str(previous_model_npz))

            # Train model
            self.train_handler.train_model(
                log_path=log_file,
                save_dir=model_folder,
                npz_save_path=new_model_npz,
            )

            print(f"Completed iteration {i+1}")