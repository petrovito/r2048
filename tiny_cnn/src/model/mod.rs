pub trait Model {
    fn forward(&self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32>;
    fn backward(&mut self, grad: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32>;
    fn update_params(&mut self, learning_rate: f32);
}

pub mod policy_model; 