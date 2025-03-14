use crate::types::{MoveDirection, Position};

/// Represents a node in the game tree
#[derive(Debug, Clone)]
pub struct GameTreeNode {
    /// The position at this node
    pub position: Position,
    /// The move that led to this position (None for the root node)
    pub move_direction: Option<MoveDirection>,
    /// The children of this node
    pub children: Vec<GameTreeNode>,
    /// The evaluation score of this node
    pub score: Option<f32>,
}

/// Creates a game tree for position evaluation
pub struct GameTreeCreator {
    /// The maximum depth to explore
    max_depth: usize,
}

impl GameTreeCreator {
    /// Creates a new GameTreeCreator with the given maximum depth
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Creates a game tree for the given position
    pub fn create_tree(&self, position: &Position) -> GameTreeNode {
        let root = GameTreeNode {
            position: position.clone(),
            move_direction: None,
            children: Vec::new(),
            score: None,
        };

        self.expand_node(root, 0)
    }

    /// Expands a node by generating its children
    fn expand_node(&self, mut node: GameTreeNode, depth: usize) -> GameTreeNode {
        // If we've reached the maximum depth or the game is over, don't expand further
        if depth >= self.max_depth || node.position.is_over() {
            return node;
        }

        // Generate children for each possible move
        for &direction in MoveDirection::all().iter() {
            let new_position = node.position.calc_move(direction);
            
            // Only add the child if the move changes the position
            if new_position.is_ok() {
                let child = GameTreeNode {
                    position: new_position.unwrap(),
                    move_direction: Some(direction),
                    children: Vec::new(),
                    score: None,
                };
                
                // Recursively expand the child
                let expanded_child = self.expand_node(child, depth + 1);
                node.children.push(expanded_child);
            }
        }

        node
    }
} 