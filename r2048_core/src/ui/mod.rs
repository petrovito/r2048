pub mod console_ui;
pub mod noop_ui;

use crate::types::{MoveDirection, Position};

/// Interface for UI interactions
pub trait UIHandler {
    /// Displays the current position
    fn show_position(&self, position: &Position);
    
    /// Displays a move
    fn show_move(&self, direction: MoveDirection);
    
    /// Displays the game over message
    fn show_game_over(&self, score: u32, highest_tile: u32);
    
    /// Displays the current score
    fn show_score(&self, score: u32);
}

pub use console_ui::ConsoleUI;
pub use noop_ui::NoopUI;
