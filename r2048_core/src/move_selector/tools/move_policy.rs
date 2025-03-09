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

/// An exploration-focused move policy that balances exploration and exploitation
pub struct UCBMovePolicy {
    /// The exploration parameter
    exploration_weight: f32,
}

impl UCBMovePolicy {
    /// Creates a new UCBMovePolicy with the given exploration weight
    pub fn new(exploration_weight: f32) -> Self {
        Self { exploration_weight }
    }
    
    /// Calculates the UCB score for a node
    fn ucb_score(&self, node_score: f32, node_visits: usize, parent_visits: usize) -> f32 {
        let exploitation = node_score;
        let exploration = (2.0 * (parent_visits as f32).ln() / (node_visits as f32)).sqrt();
        
        exploitation + self.exploration_weight * exploration
    }
}

impl MovePolicy for UCBMovePolicy {
    fn select_move(&self, root: &GameTreeNode) -> Option<MoveDirection> {
        // For simplicity, we'll just use the greedy approach here
        // In a real implementation, we would track visit counts and use UCB
        GreedyMovePolicy::new().select_move(root)
    }
} 