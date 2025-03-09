use std::fs;
use std::path::Path;

use log::info;
use serde::{Deserialize, Serialize};
use serde_yaml;

use crate::game_logger::{GameLogger, GameLoggerConfig};
use crate::game_play::GamePlayer;
use crate::move_selector::{MoveSelector, RandomSelector};
use crate::types::Game;
use crate::ui::{ConsoleUI, NoopUI, UIHandler};

/// Configuration for component selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvSpecs {
    /// The type of move selector to use
    pub selector_type: String,
    /// The type of UI handler to use
    pub ui_type: String,
    /// Whether to use verbose logging
    pub verbose_logging: bool,
    /// The number of games to play
    pub game_count: usize,
}

impl Default for EnvSpecs {
    fn default() -> Self {
        Self {
            selector_type: "random".to_string(),
            ui_type: "console".to_string(),
            verbose_logging: false,
            game_count: 1,
        }
    }
}

/// Acts as the dependency injector
pub struct Environment {
    specs: EnvSpecs,
    game: Game,
    move_selector: Box<dyn MoveSelector>,
    ui_handler: Box<dyn UIHandler>,
    game_logger: GameLogger,
}

impl Environment {
    /// Creates a new Environment with the given specifications
    pub fn new(specs: EnvSpecs) -> Self {
        let game = Game::new();
        let move_selector = Self::create_move_selector(&specs);
        let ui_handler = Self::create_ui_handler(&specs);
        let game_logger = Self::create_game_logger(&specs);

        Self {
            specs,
            game,
            move_selector,
            ui_handler,
            game_logger,
        }
    }

    /// Creates a new Environment from a YAML configuration file
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let specs: EnvSpecs = serde_yaml::from_str(&content)?;
        Ok(Self::new(specs))
    }

    /// Creates a new Environment with default specifications
    pub fn default() -> Self {
        Self::new(EnvSpecs::default())
    }

    /// Creates a move selector based on the specifications
    fn create_move_selector(specs: &EnvSpecs) -> Box<dyn MoveSelector> {
        match specs.selector_type.as_str() {
            "random" => Box::new(RandomSelector::new()),
            // Add other selectors here
            _ => Box::new(RandomSelector::new()),
        }
    }

    /// Creates a UI handler based on the specifications
    fn create_ui_handler(specs: &EnvSpecs) -> Box<dyn UIHandler> {
        match specs.ui_type.as_str() {
            "console" => Box::new(ConsoleUI::new()),
            "noop" => Box::new(NoopUI::new()),
            // Add other UI handlers here
            _ => Box::new(ConsoleUI::new()),
        }
    }

    /// Creates a game logger based on the specifications
    fn create_game_logger(specs: &EnvSpecs) -> GameLogger {
        let config = GameLoggerConfig {
            verbose: specs.verbose_logging,
        };
        GameLogger::with_config(config)
    }

    /// Plays the specified number of games
    pub fn play_games(&mut self) {
        info!("Playing {} games", self.specs.game_count);

        for i in 0..self.specs.game_count {
            info!("Starting game {}", i + 1);
            let mut player = GamePlayer::new(
                self.game.clone(),
                self.clone_move_selector(),
                self.clone_ui_handler(),
                self.game_logger.clone(),
            );
            player.play_a_game();
        }
    }

    /// Clones the move selector
    fn clone_move_selector(&self) -> Box<dyn MoveSelector> {
        // For now, we only have RandomSelector, so we can just create a new one
        Box::new(RandomSelector::new())
    }

    /// Clones the UI handler
    fn clone_ui_handler(&self) -> Box<dyn UIHandler> {
        // Create a new UI handler based on the specs
        match self.specs.ui_type.as_str() {
            "console" => Box::new(ConsoleUI::new()),
            "noop" => Box::new(NoopUI::new()),
            _ => Box::new(ConsoleUI::new()),
        }
    }
} 