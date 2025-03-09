use serde::{Deserialize, Serialize};

use super::position::Position;

/// Tracks the history of positions in a game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameHistory {
    positions: Vec<Position>,
}

impl GameHistory {
    /// Creates a new empty game history
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
        }
    }

    /// Adds a position to the history
    pub fn push(&mut self, position: Position) {
        self.positions.push(position);
    }

    /// Returns the number of positions in the history
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns true if the history is empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns a reference to the positions in the history
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// Returns the last position in the history, if any
    pub fn last(&self) -> Option<&Position> {
        self.positions.last()
    }

    /// Clears the history
    pub fn clear(&mut self) {
        self.positions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_history() {
        let mut history = GameHistory::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert!(history.last().is_none());

        let pos1 = Position::new();
        history.push(pos1.clone());
        assert!(!history.is_empty());
        assert_eq!(history.len(), 1);
        assert_eq!(history.last(), Some(&pos1));

        let mut pos2 = Position::new();
        pos2.set(0, 0, 2);
        history.push(pos2.clone());
        assert_eq!(history.len(), 2);
        assert_eq!(history.last(), Some(&pos2));
        assert_eq!(history.positions().len(), 2);

        history.clear();
        assert!(history.is_empty());
    }
} 