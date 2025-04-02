use clap::Parser;
use env_logger::Env;
use log::info;
use std::path::PathBuf;

use r2048_core::environment::{Environment, EnvSpecs};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of games to play
    #[arg(short, long, default_value_t = 1)]
    games: usize,

    /// Path to the game logs file
    #[arg(short, long, default_value = "game_data/game_logs.txt")]
    log_path: String,

    /// Type of move selector (e.g., random, policy)
    #[arg(short, long, default_value = "random")]
    selector: String,

    /// Path to the policy model (required if selector is 'policy')
    #[arg(short, long)]
    model_path: Option<PathBuf>,

    /// Type of UI handler (e.g., console, noop)
    #[arg(short, long, default_value = "console")]
    ui: String,
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    info!("Starting r2048");

    let args = Args::parse();

    // Validate that model path is provided when using policy selector
    if args.selector == "policy" && args.model_path.is_none() {
        panic!("Model path must be provided when using policy selector");
    }

    let specs = EnvSpecs {
        game_count: args.games,
        game_logs_path: args.log_path,
        selector_type: args.selector,
        policy_model_path: args.model_path,
        ui_type: args.ui,
    };

    let env = Environment::new(specs);

    for _ in 0..args.games {
        env.game_player.borrow_mut().play_a_game();
    }

    info!("r2048 completed");
}