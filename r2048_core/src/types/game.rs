use super::game_history::GameHistory;
use super::move_direction::MoveDirection;
use super::position::Position;
use super::errors::{IllegalMoveError, IllegalStateError, GameError};
use super::move_maker::MoveMaker;

/// Represents a game of 2048
#[derive(Debug, Clone)]
pub struct Game {
    /// The current position of the game
    current_position: Position,
    /// The history of positions in the game
    history: GameHistory,
    /// The move maker that handles move mechanics
    move_maker: MoveMaker,
    /// Whether the game has started
    game_started: bool,
    /// Whether the game is over
    game_over: bool,
}

impl Game {
    /// Creates a new game
    pub fn new() -> Self {
        Self {
            current_position: Position::new(),
            history: GameHistory::new(),
            move_maker: MoveMaker::new(),
            game_started: false,
            game_over: false,
        }
    }

    /// Creates a new game with a custom move maker
    pub fn with_move_maker(move_maker: MoveMaker) -> Self {
        Self {
            current_position: Position::new(),
            history: GameHistory::new(),
            move_maker,
            game_started: false,
            game_over: false,
        }
    }

    /// Starts a new game
    pub fn start_game(&mut self) -> Result<(), IllegalStateError> {
        if self.game_started {
            return Err(IllegalStateError::new("Game already started".to_string()));
        }

        self.move_maker.initialize_board(&mut self.current_position);
        self.game_started = true;
        Ok(())
    }

    /// Executes a move and updates the game state
    pub fn do_move(&mut self, direction: MoveDirection) -> Result<(), GameError> {
        if !self.game_started {
            return Err(GameError::IllegalState(IllegalStateError::new("Game not started".to_string())));
        }

        if self.game_over {
            return Err(GameError::IllegalMove(IllegalMoveError::new(direction).game_over_reason()));
        }

        // Save the old position to history
        self.history.push(self.current_position.clone());

        // Make the move
        self.current_position = self.move_maker.make_move(&self.current_position, direction)?;

        // Check if the game is over
        self.game_over = self.current_position.is_over();

        Ok(())
    }

    /// Returns a reference to the current position
    pub fn current_position(&self) -> &Position {
        &self.current_position
    }

    /// Returns a mutable reference to the current position
    pub fn current_position_mut(&mut self) -> &mut Position {
        &mut self.current_position
    }

    /// Returns a reference to the game history
    pub fn history(&self) -> &GameHistory {
        &self.history
    }

    /// Returns whether the game is over
    pub fn is_over(&self) -> bool {
        self.game_over
    }

    /// Returns the highest tile value on the board
    pub fn highest_tile(&self) -> u32 {
        self.current_position.highest_tile()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_do_move() {
        let mut game = Game::new();
        
        // Set up a specific position
        let mut position = Position::new();
        position.set(0, 0, 2);
        position.set(0, 1, 2);
        game.current_position = position;
        game.game_started = true;
        
        // Do a move
        assert!(game.do_move(MoveDirection::Left).is_ok());
        
        // Should have merged the 2s and added a new random number
        assert_eq!(game.current_position().get(0, 0), 4);
        assert_eq!(game.current_position().empty_cells(), 14);
        assert_eq!(game.history().len(), 1);
    }

    #[test]
    fn test_invalid_move() {
        let mut game = Game::new();
        
        // Set up a position where left move is invalid
        let mut position = Position::new();
        position.set(0, 0, 2);
        position.set(1, 0, 4);
        *game.current_position_mut() = position;
        
        // Try an invalid move
        assert!(game.do_move(MoveDirection::Left).is_err());
        
        // State should not have changed
        assert_eq!(game.current_position().get(0, 0), 2);
        assert_eq!(game.current_position().get(1, 0), 4);
        assert_eq!(game.history().len(), 0);
    }

    #[test]
    fn test_game_over() {
        let mut game = Game::new();
        
        // Set up a position that is one move away from game over
        let grid = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 0],
        ];
        *game.current_position_mut() = Position::with_grid(grid);
        
        let move_maker = MoveMaker::new();
        game.move_maker = move_maker;
        game.game_started = true;
        
        // Do a move
        assert!(game.do_move(MoveDirection::Right).is_ok());
        
        // Game should be over
        assert!(game.is_over());
    }
} 