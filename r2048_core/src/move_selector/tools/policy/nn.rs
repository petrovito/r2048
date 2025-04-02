use std::{cell::RefCell, path::Path};
use tiny_cnn::{utils::run_inference, ModelLoader, PolicyModel};
use crate::{
    move_selector::tools::policy::{MovePolicy, MovePolicyProvider},
    types::{Position, MoveDirection},
    types::errors::GameError,
};

/// A policy provider that uses a neural network to predict move probabilities
pub struct NNPolicyProvider {
    model: RefCell<PolicyModel>,
}

impl NNPolicyProvider {
    /// Creates a new NNPolicyProvider by loading a model from the given path
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, ()> {
        let mut loader = ModelLoader::new();
        let model = loader.load_from_npz(model_path).unwrap();
        Ok(Self { model: RefCell::new(model) })
    }

    /// Converts a game board into the format expected by the neural network
    fn board_to_input(&self, position: &Position) -> [f32; 16] {
        let mut input = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                input[i * 4 + j] = position.get(i, j) as f32;
            }
        }
        input
    }
}

impl MovePolicyProvider for NNPolicyProvider {
    fn get_policy(&self, position: &Position) -> Result<MovePolicy, GameError> {
        let input = self.board_to_input(position);
        let output = run_inference(&input, &mut self.model.borrow_mut());
        
        // Convert output to policy
        let policy = MovePolicy::new([
            output[MoveDirection::Up as usize], // up
            output[MoveDirection::Right as usize], // right
            output[MoveDirection::Down as usize], // down
            output[MoveDirection::Left as usize], // left
        ]);
        
        Ok(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Position;

    #[test]
    fn test_nn_policy_provider() {
        let provider = NNPolicyProvider::new("models/policy_model.npz").unwrap();
        let mut position = Position::new();
        position.set(0, 0, 2);
        position.set(0, 1, 2);
        
        let policy = provider.get_policy(&position).unwrap();
        
        // Check that probabilities sum to 1
        let sum: f32 = MoveDirection::iter().map(|d| policy.get(d)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that all probabilities are between 0 and 1
        for direction in MoveDirection::iter() {
            let prob = policy.get(direction);
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }
} 