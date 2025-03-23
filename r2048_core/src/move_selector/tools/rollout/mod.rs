use crate::types::{MoveDirection, Position};
use crate::types::errors::GameError;

mod policy_rollout;
pub use policy_rollout::PolicyRollout;

/// The result of a rollout
pub struct RolloutResult {
    pub score: f32,
    // Add additional fields as necessary
}

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

/// Trait for running rollouts
pub trait RolloutRunner {
    fn rollout(&self, position: &Position) -> RolloutResult;
}

/// Trait for move policy providers
pub trait MovePolicyProvider {
    /// Returns a policy mapping MoveDirection to percentages that add up to 1
    fn get_policy(&self, position: &Position) -> Result<MovePolicy, GameError>;
} 