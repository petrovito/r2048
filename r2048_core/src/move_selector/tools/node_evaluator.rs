use crate::move_selector::tools::game_tree_creator::GameTreeNode;
use crate::types::Position;

/// Interface for node evaluation strategies
pub trait NodeEvaluator {
    /// Evaluates a position and returns a score
    fn evaluate_position(&self, position: &Position) -> f32;
    
    /// Evaluates a game tree node and its children
    fn evaluate_node(&self, node: &mut GameTreeNode) {
        // First, evaluate all children recursively
        for child in &mut node.children {
            self.evaluate_node(child);
        }
        
        // If the node has no children, evaluate the position directly
        if node.children.is_empty() {
            node.score = Some(self.evaluate_position(&node.position));
        } else {
            // Otherwise, the node's score is the maximum of its children's scores
            // (assuming we're maximizing our score)
            let max_score = node.children
                .iter()
                .filter_map(|child| child.score)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            
            node.score = Some(max_score);
        }
    }
}

/// A simple heuristic-based node evaluator
pub struct SimpleNodeEvaluator;

impl SimpleNodeEvaluator {
    /// Creates a new SimpleNodeEvaluator
    pub fn new() -> Self {
        Self
    }
}

impl NodeEvaluator for SimpleNodeEvaluator {
    fn evaluate_position(&self, position: &Position) -> f32 {
        // A simple heuristic: score is the sum of all tiles
        position.score() as f32
    }
} 