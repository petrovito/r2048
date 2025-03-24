use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::ThreadRng;
use rand::thread_rng;
use crate::types::{MoveDirection, Position, MoveMaker};
use crate::types::errors::{GameError, IllegalStateError};
use std::cell::RefCell;

use super::{MovePolicy, MovePolicyProvider, RolloutResult, RolloutRunner};

/// RolloutRunner implementation that uses a MovePolicyProvider
pub struct PolicyRollout<M: MovePolicyProvider> {
    move_policy_provider: M,
    num_rollouts: usize,
    max_depth: usize,
    rng: RefCell<ThreadRng>,
    move_maker: MoveMaker,
}

impl<M: MovePolicyProvider> PolicyRollout<M> {
    /// Creates a new PolicyRollout with the given MovePolicyProvider
    pub fn new(move_policy_provider: M, move_maker: MoveMaker) -> Self {
        Self { 
            move_policy_provider,
            num_rollouts: 100,    // Default number of rollouts
            max_depth: 200,       // Default max depth of each rollout
            rng: RefCell::new(thread_rng()),
            move_maker,
        }
    }

    /// Creates a new PolicyRollout with custom parameters
    pub fn with_params(move_policy_provider: M, num_rollouts: usize, max_depth: usize, move_maker: MoveMaker) -> Self {
        Self {
            move_policy_provider,
            num_rollouts,
            max_depth,
            rng: RefCell::new(thread_rng()),
            move_maker,
        }
    }
}

impl<M: MovePolicyProvider> RolloutRunner for PolicyRollout<M> {
    fn rollout(&self, position: &Position) -> RolloutResult {
        let mut total_score = 0.0;
        let mut successful_rollouts = 0;

        for _ in 0..self.num_rollouts {
            match self.single_rollout(position) {
                Ok(score) => {
                    total_score += score as f32;
                    successful_rollouts += 1;
                }
                Err(_) => continue, // Skip failed rollouts
            }
        }

        // If all rollouts failed, return a very low score
        if successful_rollouts == 0 {
            return RolloutResult { score: -1000.0 };
        }

        RolloutResult {
            score: total_score / successful_rollouts as f32,
        }
    }
}

impl<M: MovePolicyProvider> PolicyRollout<M> {
    fn single_rollout(&self, start_position: &Position) -> Result<u32, GameError> {
        let mut position = start_position.clone();
        let mut depth = 0;

        // Continue until game is over or we reach max depth
        while !position.is_over() && depth < self.max_depth {
            // Get move policy for current position
            let policy = self.move_policy_provider.get_policy(&position)?;
            
            // Choose move according to policy
            let selected_move = self.select_move_from_policy(&policy)?;
            
            // Make the move
            position = match self.move_maker.make_move(&position, selected_move) {
                Ok(new_position) => new_position,
                Err(err) => return Err(err.into()),
            };
            
            depth += 1;
        }

        Ok(position.score())
    }

    fn select_move_from_policy(&self, policy: &MovePolicy) -> Result<MoveDirection, GameError> {
        // Get all directions and their probabilities
        let directions = MoveDirection::all();
        let weights: Vec<f32> = directions.iter()
            .map(|&dir| policy.get(dir))
            .collect();
        
        // Check if all weights are zero
        if weights.iter().all(|&w| w == 0.0) {
            return Err(IllegalStateError::from_str("No valid moves in policy (all probabilities are zero)").into());
        }
        
        // Create a weighted distribution
        let dist = match WeightedIndex::new(&weights) {
            Ok(dist) => dist,
            Err(_) => return Err(IllegalStateError::from_str("Failed to create weighted distribution").into()),
        };
        
        // Select a random direction based on the weights using the class member RNG
        let selected_index = dist.sample(&mut *self.rng.borrow_mut());
        
        Ok(directions[selected_index])
    }
} 