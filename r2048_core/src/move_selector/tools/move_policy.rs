use crate::move_selector::tools::game_tree_creator::GameTreeNode;
use crate::types::MoveDirection;

/// Interface for move selection policies
pub trait MovePolicy {
    /// Selects the best move from a game tree
    fn select_move(&self, root: &GameTreeNode) -> Option<MoveDirection>;
}

/// A greedy move policy that selects the move with the highest score
pub struct GreedyMovePolicy;

impl GreedyMovePolicy {
    /// Creates a new GreedyMovePolicy
    pub fn new() -> Self {
        Self
    }
}

impl MovePolicy for GreedyMovePolicy {
    fn select_move(&self, root: &GameTreeNode) -> Option<MoveDirection> {
        // Find the child with the highest score
        root.children
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|best_child| best_child.move_direction)
    }
}
