use crate::types::{MoveDirection, Position};
use crate::ui::UIHandler;

/// A null implementation of UIHandler for headless operation
#[derive(Debug, Clone)]
pub struct NoopUI;

impl NoopUI {
    /// Creates a new NoopUI
    pub fn new() -> Self {
        Self
    }
}

impl UIHandler for NoopUI {
    fn show_position(&self, _position: &Position) {
        // No operation
    }
    
    fn show_move(&self, _direction: MoveDirection) {
        // No operation
    }
    
    fn show_game_over(&self, _score: u32, _highest_tile: u32) {
        // No operation
    }
    
    fn show_score(&self, _score: u32) {
        // No operation
    }
} 