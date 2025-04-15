import subprocess
from pathlib import Path

class SimulatorRunner:
    def __init__(self, cargo_manifest_path: str):
        self.cargo_manifest_path = cargo_manifest_path
        self.selector = "policy"

    def run_simulation(self, games: int, log_path: Path, model_path: str):
        """Run the Cargo simulation."""
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cargo_cmd = [
            "cargo", "run",
            "--manifest-path", self.cargo_manifest_path,
            "--",
            "--games", str(games),
            "--log-path", str(log_path),
            "--selector", self.selector,
            "--model-path", model_path
        ]

        print(f"Running Cargo simulation with log path: {log_path}")
        result = subprocess.run(cargo_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Cargo simulation failed with return code {result.returncode}")

        print("Simulation completed successfully.")