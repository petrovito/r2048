use std::error::Error;
use std::fmt;

use super::move_direction::MoveDirection;

#[derive(Debug)]
pub struct IllegalMoveError {
    direction: MoveDirection,
    game_over: bool,
}

impl IllegalMoveError {
    pub fn new(direction: MoveDirection) -> Self {
        Self { direction, game_over: false }
    }

    pub fn game_over_reason(mut self) -> Self {
        self.game_over = true;
        self
    }
}


impl fmt::Display for IllegalMoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.game_over {
            write!(f, "Game over")
        } else {
            write!(f, "Illegal move: {:?}", self.direction)
        }
    }
}

impl Error for IllegalMoveError {}

#[derive(Debug)]
pub struct IllegalStateError {  
    message: String,
}

impl IllegalStateError {
    pub fn new(message: String) -> Self {
        Self { message }
    }

    pub fn from_str(message: &str) -> Self {
        Self { message: message.to_string() }
    }
}

impl fmt::Display for IllegalStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for IllegalStateError {}

#[derive(Debug)]
pub enum GameError {
    IllegalMove(IllegalMoveError),
    IllegalState(IllegalStateError),
}


impl Error for GameError {}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::IllegalMove(e) => write!(f, "{}", e),
            GameError::IllegalState(e) => write!(f, "{}", e),
        }
    }
}

impl From<IllegalMoveError> for GameError {
    fn from(e: IllegalMoveError) -> Self {
        GameError::IllegalMove(e)
    }
}

impl From<IllegalStateError> for GameError {
    fn from(e: IllegalStateError) -> Self {
        GameError::IllegalState(e)
    }
}



