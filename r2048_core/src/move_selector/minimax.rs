use crate::move_selector::{MoveSelector, tools::{GameTreeCreator, NodeEvaluator, SimpleNodeEvaluator}};
use crate::types::{MoveDirection, Position, GameError};

/// A minimax-based move selector
/// Creates a game tree of depth max_depth then evaluates the leaf nodes using the node_evaluator
/// Then works back up to the root node using a modified minimax algorithm:
/// There are 2 "players": the real player and the chance player
/// Takes max at a chance node, and takes average (EV) at a real player node
pub struct MinimaxSelector {
    tree_creator: GameTreeCreator,
    node_evaluator: Box<dyn NodeEvaluator>,
}

impl MinimaxSelector {
    /// Creates a new MinimaxSelector with default components
    pub fn new(max_depth: usize) -> Self {
        Self {
            tree_creator: GameTreeCreator::new(max_depth),
            node_evaluator: Box::new(SimpleNodeEvaluator::new()),
        }
    }
    
    /// Creates a new MinimaxSelector with custom components
    pub fn with_components(
        tree_creator: GameTreeCreator,
        node_evaluator: Box<dyn NodeEvaluator>,
    ) -> Self {
        Self {
            tree_creator,
            node_evaluator,
        }
    }
}

impl MoveSelector for MinimaxSelector {
    fn make_move(&self, position: &Position) -> Result<MoveDirection, GameError> {
        // Create the game tree
        let mut root = self.tree_creator.create_tree(position);
        // Evaluate leaf nodes
        //TODO
        return Ok(MoveDirection::Up);
    }
} 