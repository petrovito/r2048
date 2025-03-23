pub mod random;
pub mod minimax;
pub mod mcts;
pub mod tools;

use crate::types::{MoveDirection, Position, GameError};

/// Interface for move selection strategies
pub trait MoveSelector {
    /// Returns the selected move for the given position
    fn make_move(&self, position: &Position) -> Result<MoveDirection, GameError>;
}

pub use random::RandomSelector;
pub use minimax::MinimaxSelector;
pub use mcts::MCTSSelector; 