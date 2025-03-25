use std::path::Path;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::model::Model;

pub trait ModelPersistence {
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path) -> Result<Self> where Self: Sized;
}

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize),
    pub learning_rate: f32,
    pub batch_size: usize,
}

impl ModelConfig {
    pub fn new(
        input_shape: (usize, usize),
        output_shape: (usize, usize),
        learning_rate: f32,
        batch_size: usize,
    ) -> Self {
        Self {
            input_shape,
            output_shape,
            learning_rate,
            batch_size,
        }
    }
} 