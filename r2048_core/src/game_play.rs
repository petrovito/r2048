use std::cell::RefCell;
use std::rc::Rc;

use log::{info, error};

use crate::move_selector::MoveSelector;
use crate::types::move_maker::MoveMaker;
use crate::types::Game;
use crate::types::GameError;
use crate::game_logger::GameLogger;
/// Coordinates game execution
pub struct GamePlayer {
    game: Game,
    move_selector: Rc<dyn MoveSelector>,
    game_logger: Rc<RefCell<GameLogger>>,
}

impl GamePlayer {
    /// Creates a new GamePlayer
    pub fn new(
        move_maker: MoveMaker,
        move_selector: Rc<dyn MoveSelector>,
        game_logger: Rc<RefCell<GameLogger>>,
    ) -> Self {
        Self {
            game: Game::with_move_maker(move_maker),
            move_selector,
            game_logger,
        }
    }

    /// Plays a complete game
    pub fn play_a_game(&mut self) {
        let _ = self.game.start_game();

        while !self.game.is_over() {
            let _ = self.play_a_move();
        }

        if let Ok(_) = self.game_logger.borrow_mut().log_game(&self.game) {
            info!("Game logged successfully. Score: {}", self.game.current_position().score());
        } else {
            error!("Failed to log game");
        }
    }

    /// Plays a single move
    pub fn play_a_move(&mut self) -> Result<(), GameError> {
        let direction = self.move_selector.select_move(self.game.current_position())?;
        self.game.do_move(direction)?;
        Ok(())
    }

    /// Returns a reference to the game
    pub fn game(&self) -> &Game {
        &self.game
    }

    /// Returns a mutable reference to the game
    pub fn game_mut(&mut self) -> &mut Game {
        &mut self.game
    }
} 