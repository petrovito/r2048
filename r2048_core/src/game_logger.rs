use log::{info, debug};
use serde::{Deserialize, Serialize};

use crate::types::Game;

/// Configuration for the game logger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameLoggerConfig {
    /// Whether to log detailed game information
    pub verbose: bool,
}

impl Default for GameLoggerConfig {
    fn default() -> Self {
        Self {
            verbose: false,
        }
    }
}

/// Handles game logging
#[derive(Debug, Clone)]
pub struct GameLogger {
    config: GameLoggerConfig,
}

impl GameLogger {
    /// Creates a new GameLogger with default configuration
    pub fn new() -> Self {
        Self {
            config: GameLoggerConfig::default(),
        }
    }

    /// Creates a new GameLogger with custom configuration
    pub fn with_config(config: GameLoggerConfig) -> Self {
        Self {
            config,
        }
    }

    /// Logs a complete game
    pub fn log_game(&self, game: &Game) {
        info!(
            "Game completed. Highest tile: {}",
            game.highest_tile()
        );

        if self.config.verbose {
            debug!("Game history length: {}", game.history().len());
            debug!("Final position: {:?}", game.current_position());
        }
    }
} 