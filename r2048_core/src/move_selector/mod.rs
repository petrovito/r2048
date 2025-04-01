pub mod random;
pub mod mcts;
pub mod tools;
pub mod policy_selector;

use crate::types::{MoveDirection, Position, GameError};
use crate::types::move_maker::MoveMaker;

/// Interface for move selection strategies
pub trait MoveSelector {
    /// Returns the selected move for the given position
    fn select_move(&self, position: &Position) -> Result<MoveDirection, GameError>;
    /// Set the move maker for the move selector
    fn set_move_maker(&mut self, move_maker: &MoveMaker);
}

pub use random::RandomSelector;
pub use mcts::MCTSSelector;
pub use policy_selector::PolicySelector; 