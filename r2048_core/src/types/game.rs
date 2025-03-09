use serde::{Deserialize, Serialize};

use super::game_history::GameHistory;
use super::move_direction::MoveDirection;
use super::number_popper::NumberPopper;
use super::position::Position;

/// Represents a game of 2048
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Game {
    /// The current position of the game
    current_position: Position,
    /// The history of positions in the game
    history: GameHistory,
    /// The number popper for generating random numbers
    number_popper: NumberPopper,
    /// The score of the game
    score: u32,
    /// Whether the game is over
    game_over: bool,
}

impl Game {
    /// Creates a new game
    pub fn new() -> Self {
        Self {
            current_position: Position::new(),
            history: GameHistory::new(),
            number_popper: NumberPopper::new(),
            score: 0,
            game_over: false,
        }
    }

    /// Creates a new game with a custom number popper
    pub fn with_number_popper(number_popper: NumberPopper) -> Self {
        Self {
            current_position: Position::new(),
            history: GameHistory::new(),
            number_popper,
            score: 0,
            game_over: false,
        }
    }

    /// Starts a new game
    pub fn start_game(&mut self) {
        self.current_position = Position::new();
        self.history.clear();
        self.score = 0;
        self.game_over = false;
        self.number_popper.initialize_board(&mut self.current_position);
    }

    /// Executes a move and updates the game state
    pub fn do_move(&mut self, direction: MoveDirection) -> bool {
        if self.game_over {
            return false;
        }

        let old_position = self.current_position.clone();
        let new_position = self.current_position.calc_move(direction);

        // If the position didn't change, the move is invalid
        if new_position == old_position {
            return false;
        }

        // Update the score
        let old_score = old_position.score();
        let new_score = new_position.score();
        self.score += new_score - old_score;

        // Save the old position to history
        self.history.push(old_position);

        // Update the current position
        self.current_position = new_position;

        // Add a new random number
        self.number_popper.pop_random_number(&mut self.current_position);

        // Check if the game is over
        self.game_over = self.current_position.is_over();

        true
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

    /// Returns the current score
    pub fn score(&self) -> u32 {
        self.score
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
    fn test_new_game() {
        let game = Game::new();
        assert_eq!(game.score(), 0);
        assert!(!game.is_over());
        assert!(game.history().is_empty());
    }

    #[test]
    fn test_start_game() {
        let mut game = Game::new();
        game.start_game();
        
        // Should have two random numbers on the board
        assert_eq!(game.current_position().empty_cells(), 14);
        assert_eq!(game.score(), 0);
        assert!(!game.is_over());
        assert!(game.history().is_empty());
    }

    #[test]
    fn test_do_move() {
        let mut game = Game::new();
        
        // Set up a specific position
        let mut position = Position::new();
        position.set(0, 0, 2);
        position.set(0, 1, 2);
        *game.current_position_mut() = position;
        
        // Do a move
        assert!(game.do_move(MoveDirection::Left));
        
        // Should have merged the 2s and added a new random number
        assert_eq!(game.current_position().get(0, 0), 4);
        assert_eq!(game.current_position().empty_cells(), 14);
        assert_eq!(game.history().len(), 1);
        assert_eq!(game.score(), 4);
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
        assert!(!game.do_move(MoveDirection::Left));
        
        // State should not have changed
        assert_eq!(game.current_position().get(0, 0), 2);
        assert_eq!(game.current_position().get(1, 0), 4);
        assert_eq!(game.history().len(), 0);
        assert_eq!(game.score(), 0);
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
        
        // Force a specific number to be added that will cause game over
        let mut number_popper = NumberPopper::with_probability(0.0); // Always generate 2
        game.number_popper = number_popper;
        
        // Do a move
        assert!(game.do_move(MoveDirection::Right));
        
        // Game should be over
        assert!(game.is_over());
    }
} 