use std::path::Path;
use anyhow::Result;
use ndarray::{Array2, ArrayView2};

use crate::model::Model;
use crate::persistence::ModelPersistence;

pub struct Inference<M: Model> {
    model: M,
}

impl<M: Model> Inference<M> {
    pub fn load_model(path: &Path) -> Result<Self> {
        // TODO: Implement model loading
        unimplemented!()
    }

    pub fn predict(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        self.model.forward(input)
    }

    pub fn predict_batch(&self, inputs: &ArrayView2<f32>) -> Array2<f32> {
        self.model.forward(inputs)
    }
} 