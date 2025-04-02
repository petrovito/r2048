use crate::types::{MoveDirection, Position};
use crate::types::errors::GameError;

mod nn;

pub use nn::NNPolicyProvider;

/// Represents a policy of move probabilities
pub struct MovePolicy {
    probabilities: [f32; 4],
}

impl MovePolicy {
    /// Creates a new MovePolicy with the given probabilities
    pub fn new(probabilities: [f32; 4]) -> Self {
        Self { probabilities }
    }

    /// Gets the probability for the specified direction
    pub fn get(&self, direction: MoveDirection) -> f32 {
        self.probabilities[direction as usize]
    }
}

/// Trait for move policy providers
pub trait MovePolicyProvider {
    /// Returns a policy mapping MoveDirection to percentages that add up to 1
    fn get_policy(&self, position: &Position) -> Result<MovePolicy, GameError>;
} 