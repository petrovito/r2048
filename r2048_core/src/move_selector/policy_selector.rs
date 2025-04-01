use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::ThreadRng;
use rand::thread_rng;
use std::cell::RefCell;

use crate::move_selector::{MoveSelector, tools::policy::MovePolicyProvider};
use crate::types::{MoveDirection, Position, GameError, IllegalStateError, MoveMaker};

/// A move selector that uses a policy to select moves
pub struct PolicySelector<P: MovePolicyProvider> {
    policy_provider: P,
    rng: RefCell<ThreadRng>,
    move_maker: MoveMaker,
}

impl<P: MovePolicyProvider> PolicySelector<P> {
    /// Creates a new PolicySelector with the given policy provider
    pub fn new(policy_provider: P) -> Self {
        Self {
            policy_provider,
            rng: RefCell::new(thread_rng()),
            move_maker: MoveMaker::new(),
        }
    }
}

impl<P: MovePolicyProvider> MoveSelector for PolicySelector<P> {
    fn select_move(&self, position: &Position) -> Result<MoveDirection, GameError> {
        // Get the policy for the current position
        let policy = self.policy_provider.get_policy(position)?;
        
        // Get all directions and their probabilities
        let directions = MoveDirection::all();
        let mut valid_moves = Vec::new();
        let mut weights = Vec::new();
        
        // Filter out invalid moves and collect valid ones with their weights
        for &dir in directions.iter() {
            if self.move_maker.make_move(position, dir).is_ok() {
                valid_moves.push(dir);
                weights.push(policy.get(dir));
            }
        }
        
        // Check if we have any valid moves
        if valid_moves.is_empty() {
            return Err(IllegalStateError::from_str("No valid moves").into());
        }
        
        // Check if all weights are zero
        if weights.iter().all(|&w| w == 0.0) {
            return Err(IllegalStateError::from_str("No valid moves in policy (all probabilities are zero)").into());
        }
        
        // Create a weighted distribution
        let dist = match WeightedIndex::new(&weights) {
            Ok(dist) => dist,
            Err(_) => return Err(IllegalStateError::from_str("Failed to create weighted distribution").into()),
        };
        
        // Select a random direction based on the weights
        let selected_index = dist.sample(&mut *self.rng.borrow_mut());
        
        Ok(valid_moves[selected_index])
    }

    fn set_move_maker(&mut self, move_maker: &MoveMaker) {
        self.move_maker = move_maker.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{move_selector::tools::policy::MovePolicy, types::Position};

    struct TestPolicyProvider;

    impl MovePolicyProvider for TestPolicyProvider {
        fn get_policy(&self, _position: &Position) -> Result<MovePolicy, GameError> {
            // Return a simple policy that favors up and right moves
            Ok(MovePolicy::new([0.4, 0.1, 0.1, 0.4]))  // [up, right, down, left]
        }
    }

    #[test]
    fn test_policy_selector() {
        let provider = TestPolicyProvider;
        let selector = PolicySelector::new(provider);
        let mut position = Position::new();
        
        // Initialize the board with some tiles
        position.set(0, 0, 2);
        position.set(0, 1, 2);
        
        // Test that we can get a move
        let result = selector.select_move(&position);
        println!("Result: {:?}", result);
        assert!(result.is_ok());
        
        // Test that the move is valid
        let direction = result.unwrap();
        assert!(selector.move_maker.make_move(&position, direction).is_ok());
    }
} 