use crate::game_logger::GameLogger;
use crate::move_selector::MoveSelector;
use crate::types::Game;
use crate::types::GameError;
use crate::ui::UIHandler;

/// Coordinates game execution
pub struct GamePlayer {
    game: Game,
    move_selector: Box<dyn MoveSelector>,
    ui_handler: Box<dyn UIHandler>,
    game_logger: GameLogger,
}

impl GamePlayer {
    /// Creates a new GamePlayer
    pub fn new(
        game: Game,
        move_selector: Box<dyn MoveSelector>,
        ui_handler: Box<dyn UIHandler>,
        game_logger: GameLogger,
    ) -> Self {
        Self {
            game,
            move_selector,
            ui_handler,
            game_logger,
        }
    }

    /// Plays a complete game
    pub fn play_a_game(&mut self) {
        let _ = self.game.start_game();
        self.ui_handler.show_position(self.game.current_position());

        while !self.game.is_over() {
            let _ = self.play_a_move();
        }

        //self.ui_handler.show_game_over(self.game.highest_tile());
        self.game_logger.log_game(&self.game);
    }

    /// Plays a single move
    pub fn play_a_move(&mut self) -> Result<(), GameError> {
        let direction = self.move_selector.make_move(self.game.current_position())?;
        self.game.do_move(direction)?;
        self.ui_handler.show_move(direction);
        self.ui_handler.show_position(self.game.current_position());
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