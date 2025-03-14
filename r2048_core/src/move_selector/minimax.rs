use crate::move_selector::{MoveSelector, tools::{GameTreeCreator, NodeEvaluator, SimpleNodeEvaluator, MovePolicy, GreedyMovePolicy}};
use crate::types::{MoveDirection, Position};

/// A minimax-based move selector
pub struct MinimaxSelector {
    tree_creator: GameTreeCreator,
    node_evaluator: Box<dyn NodeEvaluator>,
    move_policy: Box<dyn MovePolicy>,
}

impl MinimaxSelector {
    /// Creates a new MinimaxSelector with default components
    pub fn new(max_depth: usize) -> Self {
        Self {
            tree_creator: GameTreeCreator::new(max_depth),
            node_evaluator: Box::new(SimpleNodeEvaluator::new()),
            move_policy: Box::new(GreedyMovePolicy::new()),
        }
    }
    
    /// Creates a new MinimaxSelector with custom components
    pub fn with_components(
        tree_creator: GameTreeCreator,
        node_evaluator: Box<dyn NodeEvaluator>,
        move_policy: Box<dyn MovePolicy>,
    ) -> Self {
        Self {
            tree_creator,
            node_evaluator,
            move_policy,
        }
    }
}

impl MoveSelector for MinimaxSelector {
    fn make_move(&self, position: &Position) -> MoveDirection {
        // Create the game tree
        let mut root = self.tree_creator.create_tree(position);
        
        // Evaluate the tree
        self.node_evaluator.evaluate_node(&mut root);
        
        // Select the best move
        self.move_policy
            .select_move(&root)
            .unwrap_or_else(|| {
                // If no move is selected, fall back to a random valid move
                let valid_moves: Vec<MoveDirection> = MoveDirection::all()
                    .iter()
                    .filter(|&&dir| position.calc_move(dir).is_ok())
                    .cloned()
                    .collect();
                
                if valid_moves.is_empty() {
                    // If no valid moves, just return Up
                    MoveDirection::Up
                } else {
                    // Return the first valid move
                    valid_moves[0]
                }
            })
    }
} 