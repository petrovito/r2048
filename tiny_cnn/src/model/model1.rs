use ndarray::{Array2, ArrayView2};

use super::{Model, Bottom, Head};

pub struct Model1<B: Bottom, H: Head> {
    bottom: B,
    head: H,
}

impl<B: Bottom, H: Head> Model1<B, H> {
    pub fn new(bottom: B, head: H) -> Self {
        Self { bottom, head }
    }
}

impl<B: Bottom, H: Head> Model for Model1<B, H> {
    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let bottom_output = self.bottom.forward(input);
        self.head.forward(&bottom_output.view())
    }

    fn backward(&mut self, grad: &ArrayView2<f32>) -> Array2<f32> {
        let head_grad = self.head.backward(grad);
        self.bottom.backward(&head_grad.view())
    }

    fn update_params(&mut self, learning_rate: f32) {
        self.head.update_params(learning_rate);
        self.bottom.update_params(learning_rate);
    }
} 