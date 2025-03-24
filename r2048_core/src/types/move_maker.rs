use super::move_direction::MoveDirection;
use super::position::Position;
use super::number_popper::NumberPopper;
use super::errors::GameError;

/// Handles the mechanics of making moves in a 2048 game
#[derive(Debug, Clone)]
pub struct MoveMaker {
    number_popper: NumberPopper,
}

impl MoveMaker {
    /// Creates a new MoveMaker with a default NumberPopper
    pub fn new() -> Self {
        Self {
            number_popper: NumberPopper::new(),
        }
    }

    /// Creates a new MoveMaker with a custom NumberPopper
    pub fn with_number_popper(number_popper: NumberPopper) -> Self {
        Self {
            number_popper,
        }
    }

    /// Makes a move on the given position, returning the new position
    pub fn make_move(&self, position: &Position, direction: MoveDirection) -> Result<Position, GameError> {
        // Calculate the new position after the move
        let mut new_position = position.calc_move(direction)?;

        // Add a new random number
        self.number_popper.pop_random_number(&mut new_position);

        Ok(new_position)
    }

    /// Initializes a board with random numbers
    pub fn initialize_board(&self, position: &mut Position) {
        self.number_popper.initialize_board(position);
    }
} 