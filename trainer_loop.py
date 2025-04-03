import subprocess
import os
import argparse

def run_commands(iterations, base_log_path, start_index, python_args, cargo_args, cargo_manifest_path):
    """
    Iteratively run a Python training command and a Cargo command, incrementing the log file postscript.
    
    :param iterations: Number of iterations to run
    :param base_log_path: Base path for log files (e.g., "game_data/game_logs")
    :param start_index: Starting index for log file postscript (e.g., 3 for "game_logs3.txt")
    :param python_args: Dictionary of arguments for the Python command
    :param cargo_args: Dictionary of arguments for the Cargo command
    :param cargo_manifest_path: Path to the Cargo.toml file
    """
    for i in range(start_index, start_index + iterations):
        # Construct the log file path for this iteration
        log_file = f"{base_log_path}{i}.txt"

        # Build the Cargo command
        cargo_cmd = [
            "cargo", "run",
            "--manifest-path", cargo_manifest_path,
            "--",
            "--games", str(cargo_args["games"]),
            "--log-path", log_file,
            "--selector", cargo_args["selector"],
            "--model-path", cargo_args["model_path"]
        ]
        
        # Build the Python command
        python_cmd = [
            "python3", "-m", "training.train",
            "--log_path", log_file,
            "--num_epochs", str(python_args["num_epochs"]),
            "--npz_save_path", python_args["npz_save_path"],
            "--save_dir", python_args["save_dir"]
        ]
        
        
        print(f"\nIteration {i - start_index + 1}/{iterations}")
        print(f"Log file: {log_file}")


        # Run Cargo command
        print("Running Cargo simulation...")
        cargo_result = subprocess.run(cargo_cmd, check=False)
        if cargo_result.returncode != 0:
            print(f"Cargo command failed with return code {cargo_result.returncode}")
            break
        
        # Run Python command
        print("Running Python training...")
        python_result = subprocess.run(python_cmd, check=False)
        if python_result.returncode != 0:
            print(f"Python command failed with return code {python_result.returncode}")
            break
        
        
        print(f"Completed iteration {i - start_index + 1}")

def main():
    # Set up argument parser for customization
    parser = argparse.ArgumentParser(description="Iteratively run Python training and Cargo simulation.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--base_log_path", type=str, default="game_data/game_logs", help="Base path for log files")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for log file postscript")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for Python training")
    parser.add_argument("--npz_save_path", type=str, default="models/best.npz", help="Path to save NPZ model")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--games", type=int, default=200, help="Number of games for Cargo run")
    parser.add_argument("--selector", type=str, default="policy", help="Selector type for Cargo run")
    parser.add_argument("--model_path", type=str, default="models/best.npz", help="Model path for Cargo run")
    parser.add_argument("--cargo_manifest_path", type=str, default="r2048_core/Cargo.toml", help="Path to Cargo.toml")

    args = parser.parse_args()

    # Organize arguments into dictionaries for Python and Cargo commands
    python_args = {
        "num_epochs": args.num_epochs,
        "npz_save_path": args.npz_save_path,
        "save_dir": args.save_dir
    }
    
    cargo_args = {
        "games": args.games,
        "selector": args.selector,
        "model_path": args.model_path
    }

    # Ensure directories exist
    os.makedirs(os.path.dirname(args.base_log_path), exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Run the commands
    run_commands(
        iterations=args.iterations,
        base_log_path=args.base_log_path,
        start_index=args.start_index,
        python_args=python_args,
        cargo_args=cargo_args,
        cargo_manifest_path=args.cargo_manifest_path
    )

if __name__ == "__main__":
    main()