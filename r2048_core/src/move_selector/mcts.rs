use std::time::{Duration, Instant};

use crate::move_selector::{MoveSelector, tools::{NodeEvaluator, SimpleNodeEvaluator}};
use crate::types::{MoveDirection, Position};

/// A node in the Monte Carlo Tree Search
#[derive(Debug, Clone)]
struct MCTSNode {
    /// The position at this node
    position: Position,
    /// The move that led to this position (None for the root node)
    move_direction: Option<MoveDirection>,
    /// The children of this node
    children: Vec<MCTSNode>,
    /// The number of times this node has been visited
    visits: usize,
    /// The total score accumulated from this node
    total_score: f32,
}

impl MCTSNode {
    /// Creates a new MCTSNode
    fn new(position: Position, move_direction: Option<MoveDirection>) -> Self {
        Self {
            position,
            move_direction,
            children: Vec::new(),
            visits: 0,
            total_score: 0.0,
        }
    }
    
    /// Returns the UCB score for this node
    fn ucb_score(&self, parent_visits: usize, exploration_weight: f32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        
        let exploitation = self.total_score / self.visits as f32;
        let exploration = (2.0 * (parent_visits as f32).ln() / (self.visits as f32)).sqrt();
        
        exploitation + exploration_weight * exploration
    }
    
    /// Expands this node by generating its children
    fn expand(&mut self) {
        if !self.children.is_empty() {
            return;
        }
        
        for &direction in MoveDirection::all().iter() {
            let new_position = self.position.calc_move(direction);
            
            // Only add the child if the move changes the position
            if new_position != self.position {
                let child = MCTSNode::new(new_position, Some(direction));
                self.children.push(child);
            }
        }
    }
    
    /// Selects a child node according to the UCB formula
    fn select_child(&self, exploration_weight: f32) -> Option<usize> {
        if self.children.is_empty() {
            return None;
        }
        
        let parent_visits = self.visits;
        
        self.children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a_score = a.ucb_score(parent_visits, exploration_weight);
                let b_score = b.ucb_score(parent_visits, exploration_weight);
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
    }
}

/// A Monte Carlo Tree Search-based move selector
pub struct MCTSSelector {
    /// The maximum number of iterations to run
    max_iterations: usize,
    /// The maximum time to run
    max_time: Duration,
    /// The exploration weight for UCB
    exploration_weight: f32,
    /// The node evaluator for leaf nodes
    node_evaluator: Box<dyn NodeEvaluator>,
}

impl MCTSSelector {
    /// Creates a new MCTSSelector with default settings
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            max_time: Duration::from_secs(1),
            exploration_weight: 1.0,
            node_evaluator: Box::new(SimpleNodeEvaluator::new()),
        }
    }
    
    /// Creates a new MCTSSelector with custom settings
    pub fn with_settings(
        max_iterations: usize,
        max_time: Duration,
        exploration_weight: f32,
        node_evaluator: Box<dyn NodeEvaluator>,
    ) -> Self {
        Self {
            max_iterations,
            max_time,
            exploration_weight,
            node_evaluator,
        }
    }
    
    /// Runs the MCTS algorithm on the given position
    fn run_mcts(&self, position: &Position) -> MCTSNode {
        let mut root = MCTSNode::new(position.clone(), None);
        let start_time = Instant::now();
        
        for _ in 0..self.max_iterations {
            if start_time.elapsed() > self.max_time {
                break;
            }
            
            // Selection and expansion
            let mut current = &mut root;
            let mut path = Vec::new();
            
            while !current.position.is_over() && !current.children.is_empty() {
                path.push(current as *mut MCTSNode);
                
                if let Some(child_index) = current.select_child(self.exploration_weight) {
                    current = &mut current.children[child_index];
                } else {
                    break;
                }
            }
            
            // Expand if the node is not terminal and has no children
            if !current.position.is_over() && current.children.is_empty() {
                current.expand();
                
                // If children were added, select one of them
                if let Some(child_index) = current.select_child(self.exploration_weight) {
                    path.push(current as *mut MCTSNode);
                    current = &mut current.children[child_index];
                }
            }
            
            // Simulation (evaluate the leaf node)
            let score = self.node_evaluator.evaluate_position(&current.position);
            
            // Backpropagation
            current.visits += 1;
            current.total_score += score;
            
            for node_ptr in path {
                unsafe {
                    let node = &mut *node_ptr;
                    node.visits += 1;
                    node.total_score += score;
                }
            }
        }
        
        root
    }
}

impl MoveSelector for MCTSSelector {
    fn make_move(&self, position: &Position) -> MoveDirection {
        let root = self.run_mcts(position);
        
        // Select the child with the most visits
        root.children
            .iter()
            .max_by_key(|child| child.visits)
            .and_then(|best_child| best_child.move_direction)
            .unwrap_or_else(|| {
                // If no move is selected, fall back to a random valid move
                let valid_moves: Vec<MoveDirection> = MoveDirection::all()
                    .iter()
                    .filter(|&&dir| position.calc_move(dir) != *position)
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