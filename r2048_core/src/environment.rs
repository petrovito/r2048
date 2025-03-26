use std::cell::RefCell;
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::game_logger::GameLogger;
use crate::game_play::GamePlayer;
use crate::move_selector::{MoveSelector, RandomSelector};
use crate::types::MoveMaker;
use crate::ui::{ConsoleUI, NoopUI, UIHandler};

/// Configuration for component selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvSpecs {
    /// The type of move selector to use
    pub selector_type: String,
    /// The type of UI handler to use
    pub ui_type: String,
    /// The number of games to play
    pub game_count: usize,
    /// The path to the game logs file
    pub game_logs_path: String,
}

impl Default for EnvSpecs {
    fn default() -> Self {
        Self {
            selector_type: "random".to_string(),
            ui_type: "console".to_string(),
            game_count: 1,
            game_logs_path: "training/data/game_logs.txt".to_string(),
        }
    }
}

/// Acts as the dependency injector
#[allow(dead_code)]
pub struct Environment {
    specs: EnvSpecs,
    move_maker: MoveMaker,
    move_selector: Rc<dyn MoveSelector>,
    ui_handler: Rc<dyn UIHandler>,
    game_logger: Rc<RefCell<GameLogger>>,
    pub game_player: Rc<RefCell<GamePlayer>>,
}

impl Environment {
    /// Creates a new Environment with the given specifications
    pub fn new(specs: EnvSpecs) -> Self {
        let move_maker = MoveMaker::new();
        let move_selector = Self::create_move_selector(&specs);
        let ui_handler = Self::create_ui_handler(&specs);
        let game_logger = Self::create_game_logger(&specs);
        let game_player = Self::create_game_player(
            move_maker.clone(), move_selector.clone(), game_logger.clone());
        Self {
            specs,
            move_maker,
            move_selector,
            ui_handler,
            game_logger,
            game_player,
        }
    }


    /// Creates a new Environment with default specifications
    pub fn default() -> Self {
        Self::new(EnvSpecs::default())
    }

    /// Creates a move selector based on the specifications
    fn create_move_selector(specs: &EnvSpecs) -> Rc<dyn MoveSelector> {
        match specs.selector_type.as_str() {
            "random" => Rc::new(RandomSelector::new()),
            // Add other selectors here
            _ => Rc::new(RandomSelector::new()),
        }
    }

    /// Creates a UI handler based on the specifications
    fn create_ui_handler(specs: &EnvSpecs) -> Rc<dyn UIHandler> {
        match specs.ui_type.as_str() {
            "console" => Rc::new(ConsoleUI::new()),
            "noop" => Rc::new(NoopUI::new()),
            // Add other UI handlers here
            _ => Rc::new(ConsoleUI::new()),
        }
    }

    /// Creates a game logger based on the specifications
    fn create_game_logger(specs: &EnvSpecs) -> Rc<RefCell<GameLogger>> {
        Rc::new(RefCell::new(GameLogger::new(specs.game_logs_path.clone()).unwrap()))
    }

    /// Creates a game player based on the specifications
    fn create_game_player(
        move_maker: MoveMaker, 
        move_selector: Rc<dyn MoveSelector>, 
        game_logger: Rc<RefCell<GameLogger>>
    ) -> Rc<RefCell<GamePlayer>> {
        Rc::new(RefCell::new(GamePlayer::new(move_maker, move_selector, game_logger)))
    }
} 
