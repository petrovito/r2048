use std::path::Path;
use ndarray_npy::NpzReader;
use std::fs::File;
use anyhow::{Result, Context};

use crate::model::policy_model::PolicyModel;

pub struct ModelLoader {
}

impl ModelLoader {
    pub fn new() -> Self {
        Self {
        }
    }

    pub fn load_from_npz<P: AsRef<Path>>(&mut self, path: P) -> Result<PolicyModel> {
        let file = File::open(path).context("Failed to open npz file")?;
        let mut npz = NpzReader::new(file)?;

        // Load all parameters
        let conv1_weight = npz.by_name("conv1.weight")?;
        let conv1_bias = npz.by_name("conv1.bias")?;
        let conv2_weight = npz.by_name("conv2.weight")?;
        let conv2_bias = npz.by_name("conv2.bias")?;
        let fc1_weight = npz.by_name("fc1.weight")?;
        let fc1_bias = npz.by_name("fc1.bias")?;
        let fc2_weight = npz.by_name("fc2.weight")?;
        let fc2_bias = npz.by_name("fc2.bias")?;

        Ok(PolicyModel::new(
            conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_load_model() {
        let mut loader = ModelLoader::new();
        let mut model = loader.load_from_npz("models/proto.npz").unwrap();

        // Test forward pass with a dummy input
        let input = Array3::zeros((1, 4, 4));
        let output = model.inference(&input);
        assert_eq!(output.shape(), &[1, 4]);  // batch_size=1, num_actions=4
    }
} 