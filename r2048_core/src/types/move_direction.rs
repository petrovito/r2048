use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveDirection {
    Up,
    Down,
    Left,
    Right,
}

impl MoveDirection {
    pub fn all() -> [MoveDirection; 4] {
        [
            MoveDirection::Up,
            MoveDirection::Down,
            MoveDirection::Left,
            MoveDirection::Right,
        ]
    }

    pub fn iter() -> impl Iterator<Item = MoveDirection> {
        MoveDirection::all().into_iter()
    }
}

impl Display for MoveDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            MoveDirection::Up => "U",
            MoveDirection::Down => "D",
            MoveDirection::Left => "L",
            MoveDirection::Right => "R",
        })
    }
}



