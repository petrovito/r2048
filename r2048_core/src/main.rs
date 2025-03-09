use env_logger::Env;
use log::info;

use r2048_core::environment::Environment;

fn main() {
    // Initialize logger
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    info!("Starting r2048");

    // Create environment with default settings
    let mut env = Environment::default();

    // Play games
    env.play_games();

    info!("r2048 completed");
}
