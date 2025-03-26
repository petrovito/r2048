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

        // Write each position in the game history
        for position in game.history().positions() {
            // Convert position to comma-separated values
            let position_str = position
                .grid()
                .iter()
                .flatten()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            
            writeln!(self.file, "{}", position_str)?;
        }

        Ok(())
    }
} 