use serde::{Deserialize, Serialize};

use super::position::Position;
use super::move_direction::MoveDirection;

/// Represents a single step in the game history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStep {
    position: Position,
    direction: MoveDirection,
}

impl GameStep {
    /// Creates a new game step
    pub fn new(position: Position, direction: MoveDirection) -> Self {
        Self {
            position,
            direction,
        }
    }

    /// Gets the position at this step
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Gets the move direction at this step
    pub fn direction(&self) -> MoveDirection {
        self.direction
    }
}

/// Tracks the history of positions and moves in a game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameHistory {
    steps: Vec<GameStep>,
}

impl GameHistory {
    /// Creates a new empty game history
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
        }
    }

    /// Adds a position and move direction to the history
    pub fn push(&mut self, position: Position, direction: MoveDirection) {
        self.steps.push(GameStep::new(position, direction));
    }

    /// Returns the number of steps in the history
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the history is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns a reference to the steps in the history
    pub fn steps(&self) -> &[GameStep] {
        &self.steps
    }

    /// Returns the last step in the history, if any
    pub fn last(&self) -> Option<&GameStep> {
        self.steps.last()
    }

    /// Returns the last position in the history, if any
    pub fn last_position(&self) -> Option<&Position> {
        self.steps.last().map(|step| step.position())
    }

    /// Clears the history
    pub fn clear(&mut self) {
        self.steps.clear();
    }
}
