use rand::prelude::*;
use std::cell::RefCell;

use crate::move_selector::MoveSelector;
use crate::types::{MoveDirection, Position, GameError, IllegalStateError, MoveMaker};

/// A random move selection strategy
#[derive(Debug, Clone)]
pub struct RandomSelector {
    rng: RefCell<ThreadRng>,
    move_maker: MoveMaker,
}

impl RandomSelector {
    /// Creates a new RandomSelector
    pub fn new() -> Self {
        Self {
            rng: RefCell::new(thread_rng()),
            move_maker: MoveMaker::new(),
        }
    }
}

impl MoveSelector for RandomSelector {
    fn select_move(&self, position: &Position) -> Result<MoveDirection, GameError> {
        // Try each direction to find valid moves
        let mut valid_moves = Vec::new();
        
        for &direction in MoveDirection::all().iter() {
            if self.move_maker.make_move(position, direction).is_ok() {
                valid_moves.push(direction);
            }
        }
        
        if valid_moves.is_empty() {
            // No valid moves, return a random direction
            Err(IllegalStateError::new(String::from("No valid moves")).into())
        } else {
            // Return a random valid move
            let target_index = self.rng.borrow_mut().gen_range(0..valid_moves.len());
            Ok(valid_moves[target_index])
        }
    }

    fn set_move_maker(&mut self, move_maker: &MoveMaker) {
        self.move_maker = move_maker.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Position;

    #[test]
    fn test_random_selector() {
        let selector = RandomSelector::new();
        let position = Position::new();
        
        // Just make sure it doesn't crash
        let _move = selector.select_move(&position);
    }
    
    #[test]
    fn test_valid_moves() {
        let selector = RandomSelector::new();
        
        // Create a position with only one valid move (right)
        let mut position = Position::new();
        position.set(0, 0, 2);
        position.set(0, 1, 4);
        
        // The only valid move should be right
        let direction = selector.select_move(&position);
        assert!(direction.is_ok(), "The move should be valid");
        let direction = direction.unwrap();
        
        // Check that the move is valid
        assert!(selector.move_maker.make_move(&position, direction).is_ok(), "The move should be valid");
    }
} 