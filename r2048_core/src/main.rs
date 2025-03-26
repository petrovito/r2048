use clap::Parser;
use env_logger::Env;
use log::info;

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

    /// Type of move selector (e.g., random)
    #[arg(short, long, default_value = "random")]
    selector: String,

    /// Type of UI handler (e.g., console, noop)
    #[arg(short, long, default_value = "console")]
    ui: String,
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    info!("Starting r2048");

    let args = Args::parse();

    let specs = EnvSpecs {
        game_count: args.games,
        game_logs_path: args.log_path,
        selector_type: args.selector,
        ui_type: args.ui,
    };

    let env = Environment::new(specs);

    for _ in 0..args.games {
        env.game_player.borrow_mut().play_a_game();
    }

    info!("r2048 completed");
}