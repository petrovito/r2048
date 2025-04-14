from simulator_runner import SimulatorRunner
from train_handler import TrainHandler

class TrainManager:
    def __init__(self, simulator_runner: SimulatorRunner, train_handler: TrainHandler):
        self.simulator_runner = simulator_runner
        self.train_handler = train_handler

    def run_training_cycle(self, iterations: int, base_log_path: str, start_index: int, games: int, selector: str, model_path: str, num_epochs: int, npz_save_path: str):
        for i in range(start_index, start_index + iterations):
            log_file = f"{base_log_path}{i}.txt"

            print(f"\nIteration {i - start_index + 1}/{iterations}")
            print(f"Log file: {log_file}")

            # Run simulation
            self.simulator_runner.run_simulation(games, log_file, selector, model_path)

            # Train model
            self.train_handler.train_model(log_file, num_epochs, npz_save_path)

            print(f"Completed iteration {i - start_index + 1}")

if __name__ == "__main__":
    simulator_runner = SimulatorRunner(cargo_manifest_path="r2048_core/Cargo.toml")
    train_handler = TrainHandler(save_dir="models")

    train_manager = TrainManager(simulator_runner, train_handler)

    train_manager.run_training_cycle(
        iterations=5,
        base_log_path="game_data/game_logs",
        start_index=0,
        games=200,
        selector="policy",
        model_path="models/best.npz",
        num_epochs=50,
        npz_save_path="models/best.npz"
    )