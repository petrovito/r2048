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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_direction() {
        let direction = MoveDirection::Up;
        assert_eq!(format!("{:?}", direction), "Up");
    }
} 