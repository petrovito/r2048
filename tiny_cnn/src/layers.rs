use ndarray::{Array2, ArrayView2};
use rand::prelude::*;

pub struct Conv2D {
    weights: Array2<f32>,
    bias: Array2<f32>,
    input_shape: (usize, usize),
    output_shape: (usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize),
}

impl Conv2D {
    pub fn new(
        input_shape: (usize, usize),
        output_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Self {
        let mut rng = thread_rng();
        let weights = Array2::from_shape_fn(
            (input_shape.1, output_channels),
            |_| rng.gen_range(-0.1..0.1),
        );
        let bias = Array2::zeros((1, output_channels));

        let output_shape = (
            (input_shape.0 - kernel_size.0) / stride.0 + 1,
            (input_shape.1 - kernel_size.1) / stride.1 + 1,
        );

        Self {
            weights,
            bias,
            input_shape,
            output_shape,
            kernel_size,
            stride,
        }
    }
}

pub struct MaxPool2D {
    pool_size: (usize, usize),
    stride: (usize, usize),
}

impl MaxPool2D {
    pub fn new(pool_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            pool_size,
            stride,
        }
    }
}

pub struct Dense {
    weights: Array2<f32>,
    bias: Array2<f32>,
    input_shape: (usize, usize),
    output_shape: (usize, usize),
}

impl Dense {
    pub fn new(input_shape: (usize, usize), output_shape: (usize, usize)) -> Self {
        let mut rng = thread_rng();
        let weights = Array2::from_shape_fn(
            (input_shape.1, output_shape.1),
            |_| rng.gen_range(-0.1..0.1),
        );
        let bias = Array2::zeros((1, output_shape.1));

        Self {
            weights,
            bias,
            input_shape,
            output_shape,
        }
    }
} 