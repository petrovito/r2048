use crate::types::{MoveDirection, Position};
use crate::types::errors::GameError;
use super::policy::{MovePolicy, MovePolicyProvider};

mod policy_rollout;
pub use policy_rollout::PolicyRollout;

/// The result of a rollout
pub struct RolloutResult {
    pub score: f32,
    // Add additional fields as necessary
}

/// Trait for running rollouts
pub trait RolloutRunner {
    fn rollout(&self, position: &Position) -> RolloutResult;
} 