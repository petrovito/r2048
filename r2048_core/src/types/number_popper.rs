use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::position::Position;

/// Handles random number generation on the board
#[derive(Debug, Clone)]
pub struct NumberPopper {
    rng: ThreadRng,
    /// Probability of generating a 4 (vs a 2)
    pub four_probability: f32,
}

impl Serialize for NumberPopper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NumberPopper", 1)?;
        state.serialize_field("four_probability", &self.four_probability)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for NumberPopper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct NumberPopperDef {
            four_probability: f32,
        }

        let def = NumberPopperDef::deserialize(deserializer)?;
        Ok(NumberPopper {
            rng: thread_rng(),
            four_probability: def.four_probability,
        })
    }
}

impl NumberPopper {
    /// Creates a new NumberPopper with default settings
    pub fn new() -> Self {
        Self {
            rng: thread_rng(),
            four_probability: 0.1,
        }
    }

    /// Creates a new NumberPopper with custom settings
    pub fn with_probability(four_probability: f32) -> Self {
        Self {
            rng: thread_rng(),
            four_probability,
        }
    }

    /// Adds a random number (2 or 4) to a random empty cell
    pub fn pop_random_number(&mut self, position: &mut Position) -> bool {
        let empty_count = position.empty_cells();
        if empty_count == 0 {
            return false;
        }

        // Choose a random empty cell
        let target_index = self.rng.gen_range(0..empty_count);
        let mut current_index = 0;

        for row in 0..4 {
            for col in 0..4 {
                if position.get(row, col) == 0 {
                    if current_index == target_index {
                        // Generate a 2 or 4 based on probability
                        let value = if self.rng.gen::<f32>() < self.four_probability {
                            4
                        } else {
                            2
                        };
                        position.set(row, col, value);
                        return true;
                    }
                    current_index += 1;
                }
            }
        }

        false
    }

    /// Adds two random numbers to the board (for game initialization)
    pub fn initialize_board(&mut self, position: &mut Position) {
        self.pop_random_number(position);
        self.pop_random_number(position);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pop_random_number() {
        let mut position = Position::new();
        let mut popper = NumberPopper::new();

        // Initially all cells are empty
        assert_eq!(position.empty_cells(), 16);

        // Add a random number
        assert!(popper.pop_random_number(&mut position));
        assert_eq!(position.empty_cells(), 15);

        // The value should be either 2 or 4
        let mut found_non_zero = false;
        for row in 0..4 {
            for col in 0..4 {
                let value = position.get(row, col);
                if value != 0 {
                    found_non_zero = true;
                    assert!(value == 2 || value == 4);
                }
            }
        }
        assert!(found_non_zero);
    }

    #[test]
    fn test_initialize_board() {
        let mut position = Position::new();
        let mut popper = NumberPopper::new();

        popper.initialize_board(&mut position);

        // Should have added two numbers
        assert_eq!(position.empty_cells(), 14);

        // Count the non-zero cells
        let mut non_zero_count = 0;
        for row in 0..4 {
            for col in 0..4 {
                if position.get(row, col) != 0 {
                    non_zero_count += 1;
                }
            }
        }
        assert_eq!(non_zero_count, 2);
    }

    #[test]
    fn test_full_board() {
        // Create a full board
        let grid = [[2; 4]; 4];
        let mut position = Position::with_grid(grid);
        let mut popper = NumberPopper::new();

        // Should return false when trying to add to a full board
        assert!(!popper.pop_random_number(&mut position));
    }
} 