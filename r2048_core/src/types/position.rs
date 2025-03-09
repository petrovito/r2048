use serde::{Deserialize, Serialize};
use std::fmt;

use super::move_direction::MoveDirection;

/// Represents a 2048 game board (4x4 grid)
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    grid: [[u32; 4]; 4],
}

impl Position {
    /// Creates a new empty position
    pub fn new() -> Self {
        Self { grid: [[0; 4]; 4] }
    }

    /// Creates a position with the given grid
    pub fn with_grid(grid: [[u32; 4]; 4]) -> Self {
        Self { grid }
    }

    /// Gets the value at the specified coordinates
    pub fn get(&self, row: usize, col: usize) -> u32 {
        self.grid[row][col]
    }

    /// Sets the value at the specified coordinates
    pub fn set(&mut self, row: usize, col: usize, value: u32) {
        self.grid[row][col] = value;
    }

    /// Returns the grid
    pub fn grid(&self) -> &[[u32; 4]; 4] {
        &self.grid
    }

    /// Calculates the result of a move in the given direction
    pub fn calc_move(&self, direction: MoveDirection) -> Position {
        let mut new_position = self.clone();
        let mut moved = false;

        match direction {
            MoveDirection::Up => {
                for col in 0..4 {
                    moved |= new_position.move_column_up(col);
                }
            }
            MoveDirection::Down => {
                for col in 0..4 {
                    moved |= new_position.move_column_down(col);
                }
            }
            MoveDirection::Left => {
                for row in 0..4 {
                    moved |= new_position.move_row_left(row);
                }
            }
            MoveDirection::Right => {
                for row in 0..4 {
                    moved |= new_position.move_row_right(row);
                }
            }
        }

        if moved {
            new_position
        } else {
            self.clone()
        }
    }

    /// Checks if the game is over (no more moves possible)
    pub fn is_over(&self) -> bool {
        // If there are empty cells, the game is not over
        for row in 0..4 {
            for col in 0..4 {
                if self.grid[row][col] == 0 {
                    return false;
                }
            }
        }

        // If there are adjacent cells with the same value, the game is not over
        for row in 0..4 {
            for col in 0..3 {
                if self.grid[row][col] == self.grid[row][col + 1] {
                    return false;
                }
            }
        }

        for col in 0..4 {
            for row in 0..3 {
                if self.grid[row][col] == self.grid[row + 1][col] {
                    return false;
                }
            }
        }

        true
    }

    /// Gets the highest tile value on the board
    pub fn highest_tile(&self) -> u32 {
        let mut max = 0;
        for row in 0..4 {
            for col in 0..4 {
                max = max.max(self.grid[row][col]);
            }
        }
        max
    }

    /// Gets the score of the position (sum of all tiles)
    pub fn score(&self) -> u32 {
        let mut score = 0;
        for row in 0..4 {
            for col in 0..4 {
                score += self.grid[row][col];
            }
        }
        score
    }

    /// Gets the number of empty cells
    pub fn empty_cells(&self) -> usize {
        let mut count = 0;
        for row in 0..4 {
            for col in 0..4 {
                if self.grid[row][col] == 0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Helper method to move a column up
    fn move_column_up(&mut self, col: usize) -> bool {
        let mut moved = false;
        
        // First, compress the column (move all non-zero values up)
        for row in 0..4 {
            if self.grid[row][col] == 0 {
                // Find the next non-zero value
                for next_row in row + 1..4 {
                    if self.grid[next_row][col] != 0 {
                        self.grid[row][col] = self.grid[next_row][col];
                        self.grid[next_row][col] = 0;
                        moved = true;
                        break;
                    }
                }
            }
        }
        
        // Then, merge adjacent cells with the same value
        for row in 0..3 {
            if self.grid[row][col] != 0 && self.grid[row][col] == self.grid[row + 1][col] {
                self.grid[row][col] *= 2;
                self.grid[row + 1][col] = 0;
                moved = true;
            }
        }
        
        // Finally, compress again to fill any gaps created by merging
        for row in 0..3 {
            if self.grid[row][col] == 0 && self.grid[row + 1][col] != 0 {
                self.grid[row][col] = self.grid[row + 1][col];
                self.grid[row + 1][col] = 0;
                moved = true;
            }
        }
        
        moved
    }

    /// Helper method to move a column down
    fn move_column_down(&mut self, col: usize) -> bool {
        let mut moved = false;
        
        // First, compress the column (move all non-zero values down)
        for row in (0..4).rev() {
            if self.grid[row][col] == 0 {
                // Find the next non-zero value
                for next_row in (0..row).rev() {
                    if self.grid[next_row][col] != 0 {
                        self.grid[row][col] = self.grid[next_row][col];
                        self.grid[next_row][col] = 0;
                        moved = true;
                        break;
                    }
                }
            }
        }
        
        // Then, merge adjacent cells with the same value
        for row in (1..4).rev() {
            if self.grid[row][col] != 0 && self.grid[row][col] == self.grid[row - 1][col] {
                self.grid[row][col] *= 2;
                self.grid[row - 1][col] = 0;
                moved = true;
            }
        }
        
        // Finally, compress again to fill any gaps created by merging
        for row in (1..4).rev() {
            if self.grid[row][col] == 0 && self.grid[row - 1][col] != 0 {
                self.grid[row][col] = self.grid[row - 1][col];
                self.grid[row - 1][col] = 0;
                moved = true;
            }
        }
        
        moved
    }

    /// Helper method to move a row left
    fn move_row_left(&mut self, row: usize) -> bool {
        let mut moved = false;
        
        // First, compress the row (move all non-zero values left)
        for col in 0..4 {
            if self.grid[row][col] == 0 {
                // Find the next non-zero value
                for next_col in col + 1..4 {
                    if self.grid[row][next_col] != 0 {
                        self.grid[row][col] = self.grid[row][next_col];
                        self.grid[row][next_col] = 0;
                        moved = true;
                        break;
                    }
                }
            }
        }
        
        // Then, merge adjacent cells with the same value
        for col in 0..3 {
            if self.grid[row][col] != 0 && self.grid[row][col] == self.grid[row][col + 1] {
                self.grid[row][col] *= 2;
                self.grid[row][col + 1] = 0;
                moved = true;
            }
        }
        
        // Finally, compress again to fill any gaps created by merging
        for col in 0..3 {
            if self.grid[row][col] == 0 && self.grid[row][col + 1] != 0 {
                self.grid[row][col] = self.grid[row][col + 1];
                self.grid[row][col + 1] = 0;
                moved = true;
            }
        }
        
        moved
    }

    /// Helper method to move a row right
    fn move_row_right(&mut self, row: usize) -> bool {
        let mut moved = false;
        
        // First, compress the row (move all non-zero values right)
        for col in (0..4).rev() {
            if self.grid[row][col] == 0 {
                // Find the next non-zero value
                for next_col in (0..col).rev() {
                    if self.grid[row][next_col] != 0 {
                        self.grid[row][col] = self.grid[row][next_col];
                        self.grid[row][next_col] = 0;
                        moved = true;
                        break;
                    }
                }
            }
        }
        
        // Then, merge adjacent cells with the same value
        for col in (1..4).rev() {
            if self.grid[row][col] != 0 && self.grid[row][col] == self.grid[row][col - 1] {
                self.grid[row][col] *= 2;
                self.grid[row][col - 1] = 0;
                moved = true;
            }
        }
        
        // Finally, compress again to fill any gaps created by merging
        for col in (1..4).rev() {
            if self.grid[row][col] == 0 && self.grid[row][col - 1] != 0 {
                self.grid[row][col] = self.grid[row][col - 1];
                self.grid[row][col - 1] = 0;
                moved = true;
            }
        }
        
        moved
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "+------+------+------+------+")?;
        for row in 0..4 {
            write!(f, "|")?;
            for col in 0..4 {
                let value = self.grid[row][col];
                if value == 0 {
                    write!(f, "      |")?;
                } else {
                    write!(f, " {:<4} |", value)?;
                }
            }
            writeln!(f)?;
            writeln!(f, "+------+------+------+------+")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_position() {
        let pos = Position::new();
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(pos.get(row, col), 0);
            }
        }
    }

    #[test]
    fn test_with_grid() {
        let grid = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 65536],
        ];
        let pos = Position::with_grid(grid);
        assert_eq!(pos.grid(), &grid);
    }

    #[test]
    fn test_get_set() {
        let mut pos = Position::new();
        pos.set(1, 2, 42);
        assert_eq!(pos.get(1, 2), 42);
    }

    #[test]
    fn test_highest_tile() {
        let grid = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 65536],
        ];
        let pos = Position::with_grid(grid);
        assert_eq!(pos.highest_tile(), 65536);
    }

    #[test]
    fn test_empty_cells() {
        let mut pos = Position::new();
        assert_eq!(pos.empty_cells(), 16);
        
        pos.set(0, 0, 2);
        pos.set(1, 1, 4);
        assert_eq!(pos.empty_cells(), 14);
    }

    #[test]
    fn test_move_left() {
        let grid = [
            [2, 2, 0, 0],
            [0, 2, 2, 0],
            [2, 0, 2, 0],
            [2, 2, 2, 2],
        ];
        let pos = Position::with_grid(grid);
        let new_pos = pos.calc_move(MoveDirection::Left);
        
        let expected = [
            [4, 0, 0, 0],
            [4, 0, 0, 0],
            [4, 0, 0, 0],
            [4, 4, 0, 0],
        ];
        assert_eq!(new_pos.grid(), &expected);
    }

    #[test]
    fn test_move_right() {
        let grid = [
            [2, 2, 0, 0],
            [0, 2, 2, 0],
            [2, 0, 2, 0],
            [2, 2, 2, 2],
        ];
        let pos = Position::with_grid(grid);
        let new_pos = pos.calc_move(MoveDirection::Right);
        
        let expected = [
            [0, 0, 0, 4],
            [0, 0, 0, 4],
            [0, 0, 0, 4],
            [0, 0, 4, 4],
        ];
        assert_eq!(new_pos.grid(), &expected);
    }

    #[test]
    fn test_move_up() {
        let grid = [
            [2, 0, 2, 2],
            [2, 2, 0, 2],
            [0, 2, 2, 0],
            [0, 0, 0, 2],
        ];
        let pos = Position::with_grid(grid);
        let new_pos = pos.calc_move(MoveDirection::Up);
        
        let expected = [
            [4, 4, 4, 4],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ];
        assert_eq!(new_pos.grid(), &expected);
    }

    #[test]
    fn test_move_down() {
        let grid = [
            [2, 0, 2, 2],
            [2, 2, 0, 2],
            [0, 2, 2, 0],
            [0, 0, 0, 2],
        ];
        let pos = Position::with_grid(grid);
        let new_pos = pos.calc_move(MoveDirection::Down);
        
        let expected = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
            [4, 4, 4, 4],
        ];
        assert_eq!(new_pos.grid(), &expected);
    }

    #[test]
    fn test_is_over() {
        // Not over - has empty cells
        let grid1 = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 0],
            [8192, 16384, 32768, 65536],
        ];
        let pos1 = Position::with_grid(grid1);
        assert!(!pos1.is_over());
        
        // Not over - has adjacent same values
        let grid2 = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 32768],
        ];
        let pos2 = Position::with_grid(grid2);
        assert!(!pos2.is_over());
        
        // Game over - no moves possible
        let grid3 = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 65536],
        ];
        let pos3 = Position::with_grid(grid3);
        assert!(pos3.is_over());
    }
} 