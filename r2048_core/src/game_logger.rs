use std::fs::{File, OpenOptions};
use std::io::Write;

use crate::types::Game;

/// Handles game logging for training data
#[derive(Debug)]
pub struct GameLogger {
    file: File,
}

impl GameLogger {
    /// Creates a new GameLogger with the specified output file
    pub fn new(output_file: String) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_file)?;

        Ok(Self { file })
    }

    /// Logs a complete game to the output file
    pub fn log_game(&mut self, game: &Game) -> std::io::Result<()> {
        // Write game separator
        writeln!(self.file, "NEW GAME")?;

        // Write each step in the game history
        writeln!(self.file, "{}", game.history().steps().iter().map(|step| {
            let position_str = step
                .position()
                .grid()
                .iter()
                .flatten()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");

            format!("{} {}", position_str, step.direction())
        }).collect::<Vec<_>>().join("\n"))?;

        Ok(())
    }
} 