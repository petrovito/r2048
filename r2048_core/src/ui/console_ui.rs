use crate::types::{MoveDirection, Position};
use crate::ui::UIHandler;

/// A console implementation of UIHandler
#[derive(Debug, Clone)]
pub struct ConsoleUI;

impl ConsoleUI {
    /// Creates a new ConsoleUI
    pub fn new() -> Self {
        Self
    }
}

impl UIHandler for ConsoleUI {
    fn show_position(&self, position: &Position) {
        println!("{:?}", position);
    }
    
    fn show_move(&self, direction: MoveDirection) {
        println!("Move: {:?}", direction);
    }
    
    fn show_game_over(&self, score: u32, highest_tile: u32) {
        println!("Game over! Score: {}, Highest tile: {}", score, highest_tile);
    }
    
    fn show_score(&self, score: u32) {
        println!("Score: {}", score);
    }
} 