use ndarray::{Array1, Array2, Array3, Array4, ArrayBase, Ix1};


pub trait Layer<Input, Output> {
    fn forward(&self, input: &Input, output: &mut Output);
}

pub struct ReLULayer {
    size: usize,
}

impl ReLULayer {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    pub fn activate(&self, array: &mut Array1<f32>) {
        for x in array.iter_mut() {
            *x = x.max(0.0);
        }
    }
}

pub struct SoftmaxLayer {
    size: usize,
}

impl SoftmaxLayer {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    pub fn activate(&self, array: &mut Array1<f32>) {
        // Find max value for numerical stability
        let max_val = array.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        for x in array.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        
        // Normalize
        for x in array.iter_mut() {
            *x /= sum;
        }
    }
}

pub struct Conv2dLayer {
    out_channels: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    kernel_size: usize,
    weights: Array4<f32>, // (out_channels, in_channels, kernel_size, kernel_size)
    bias: Array1<f32>, // (out_channels)
}

impl Conv2dLayer {
    pub fn new(out_channels: usize, in_channels: usize, height: usize, width: usize, kernel_size: usize,
        weights: Array4<f32>, bias: Array1<f32>
    ) -> Self {
        Self { out_channels, in_channels, height, width, kernel_size, weights, bias }
    }
}

impl Layer<Array3<f32>, Array3<f32>> for Conv2dLayer {
    fn forward(&self, input: &Array3<f32>, output: &mut Array3<f32>) {
        //validate
        assert_eq!(input.dim(), (self.in_channels, self.height, self.width));
        assert_eq!(output.dim(), (self.out_channels, self.height - self.kernel_size + 1, self.width - self.kernel_size + 1));

        for oc in 0..self.out_channels {
            for i in 0..self.height - self.kernel_size + 1 {
                for j in 0..self.width - self.kernel_size + 1 {
                    let mut conv_sum: f32 = 0.0;

                    // Manual convolution: loop over input channels, kernel height, and kernel width
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let input_val = input[[ic, i + kh, j + kw]];
                                let kernel_val = self.weights[[oc, ic, kh, kw]];
                                conv_sum += input_val * kernel_val;
                            }
                        }
                    }

                    conv_sum += self.bias[oc];
                    output[[oc, i, j]] = conv_sum;
                }
            }
        }
    }
}

pub struct PaddedConv2dLayer {
    // padding=1
    out_channels: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    kernel_size: usize,
    weights: Array4<f32>, // (out_channels, in_channels, kernel_size, kernel_size)
    bias: Array1<f32>, // (out_channels)
}

impl PaddedConv2dLayer {
    pub fn new(out_channels: usize, in_channels: usize, height: usize, width: usize, kernel_size: usize,
        weights: Array4<f32>, bias: Array1<f32>
    ) -> Self {
        Self { out_channels, in_channels, height, width, kernel_size, weights, bias }
    }
}
impl Layer<Array3<f32>, Array3<f32>> for PaddedConv2dLayer {
    fn forward(&self, input: &Array3<f32>, output: &mut Array3<f32>) {
        // Validate
        assert_eq!(input.dim(), (self.in_channels, self.height, self.width));
        assert_eq!(output.dim(), (self.out_channels, self.height, self.width));

        for oc in 0..self.out_channels {
            for i in 0..self.height {
                for j in 0..self.width {
                    let mut conv_sum: f32 = 0.0;

                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let input_i = i as isize + kh as isize - 1;
                                let input_j = j as isize + kw as isize - 1;

                                let input_val = if input_i < 0 || input_i >= self.height as isize ||
                                                  input_j < 0 || input_j >= self.width as isize {
                                    0.0
                                } else {
                                    input[[ic, input_i as usize, input_j as usize]]
                                };

                                let kernel_val = self.weights[[oc, ic, kh, kw]];
                                conv_sum += input_val * kernel_val;
                            }
                        }
                    }

                    conv_sum += self.bias[oc];
                    output[[oc, i, j]] = conv_sum;
                }
            }
        }
    }
}

pub struct DenseLayer {
    out_features: usize,
    in_features: usize,
    weights: Array2<f32>,  // (out_features, in_features)
    bias: Array1<f32>,     // (out_features)
}

impl DenseLayer {
    pub fn new(out_features: usize, in_features: usize, weights: Array2<f32>, bias: Array1<f32>) -> Self {
        Self {
            out_features,
            in_features,
            weights,
            bias,
        }
    }
}

impl Layer<Array1<f32>, Array1<f32>> for DenseLayer {
    fn forward(&self, input: &Array1<f32>, output: &mut Array1<f32>) {
        // Validate dimensions
        assert_eq!(input.dim(), self.in_features);
        assert_eq!(output.dim(), self.out_features);

        // Matrix multiplication with bias
        for i in 0..self.out_features {
            let mut sum = 0.0;
            for j in 0..self.in_features {
                sum += input[j] * self.weights[[i, j]];
            }
            output[i] = sum + self.bias[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_conv2d_layer() {
        // Input: 1 channel, 3x3
        let input = Array3::from_shape_vec((1, 3, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();

        // Weights: 1 output channel, 1 input channel, 2x2 kernel
        let weights = Array4::from_shape_vec((1, 1, 2, 2), vec![
            1.0, 1.0,
            1.0, 1.0,
        ]).unwrap();

        // Bias: 0.0 for simplicity
        let bias = Array1::from_vec(vec![0.0]);

        // Create Conv2dLayer with 1 input channel, 1 output channel, height=3, width=3, kernel_size=2
        let layer = Conv2dLayer::new(1, 1, 3, 3, 2, weights, bias);

        // Output: 1 channel, 2x2
        let mut output = Array3::zeros((1, 2, 2));

        // Forward pass
        layer.forward(&input, &mut output);

        // Expected output:
        // Input:
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        // Kernel:
        // [[1, 1],
        //  [1, 1]]
        // Computations:
        // - Top-left: 1+2+4+5 = 12
        // - Top-right: 2+3+5+6 = 16
        // - Bottom-left: 4+5+7+8 = 24
        // - Bottom-right: 5+6+8+9 = 28
        // Output:
        // [[12, 16],
        //  [24, 28]]
        assert!((output[[0, 0, 0]] - 12.0).abs() < 1e-6);
        assert!((output[[0, 0, 1]] - 16.0).abs() < 1e-6);
        assert!((output[[0, 1, 0]] - 24.0).abs() < 1e-6);
        assert!((output[[0, 1, 1]] - 28.0).abs() < 1e-6);
    }

    #[test]
    fn test_padded_conv2d_layer() {
        // Input: 1 channel, 3x3
        let input = Array3::from_shape_vec((1, 3, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();

        // Weights: 1 output channel, 1 input channel, 2x2 kernel
        let weights = Array4::from_shape_vec((1, 1, 2, 2), vec![
            1.0, 1.0,
            1.0, 1.0,
        ]).unwrap();

        // Bias: 0.0 for simplicity
        let bias = Array1::from_vec(vec![0.0]);

        // Create PaddedConv2dLayer with 1 input channel, 1 output channel, height=3, width=3, kernel_size=2
        let layer = PaddedConv2dLayer::new(1, 1, 3, 3, 2, weights, bias);

        // Output: 1 channel, 3x3
        let mut output = Array3::zeros((1, 3, 3));

        // Forward pass
        layer.forward(&input, &mut output);

        // Expected output:
        // Input (with padding=1, zeros outside):
        // [[0, 0, 0, 0, 0],
        //  [0, 1, 2, 3, 0],
        //  [0, 4, 5, 6, 0],
        //  [0, 7, 8, 9, 0],
        //  [0, 0, 0, 0, 0]]
        // Kernel:
        // [[1, 1],
        //  [1, 1]]
        // Computations:
        // - (0,0): 0+0+0+1 = 1
        // - (0,1): 0+0+1+2 = 3
        // - (0,2): 0+0+2+3 = 5
        // - (1,0): 0+1+0+4 = 5
        // - (1,1): 1+2+4+5 = 12
        // - (1,2): 2+3+5+6 = 16
        // - (2,0): 0+4+0+7 = 11
        // - (2,1): 4+5+7+8 = 24
        // - (2,2): 5+6+8+9 = 28
        // Output:
        // [[1, 3, 5],
        //  [5, 12, 16],
        //  [11, 24, 28]]
        assert!((output[[0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((output[[0, 0, 1]] - 3.0).abs() < 1e-6);
        assert!((output[[0, 0, 2]] - 5.0).abs() < 1e-6);
        assert!((output[[0, 1, 0]] - 5.0).abs() < 1e-6);
        assert!((output[[0, 1, 1]] - 12.0).abs() < 1e-6);
        assert!((output[[0, 1, 2]] - 16.0).abs() < 1e-6);
        assert!((output[[0, 2, 0]] - 11.0).abs() < 1e-6);
        assert!((output[[0, 2, 1]] - 24.0).abs() < 1e-6);
        assert!((output[[0, 2, 2]] - 28.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_layer() {
        // Create a simple dense layer with 2 input features and 3 output features
        let weights = Array2::from_shape_vec((3, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]).unwrap();
        let bias = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        
        let layer = DenseLayer::new(3, 2, weights, bias);
        
        // Test input
        let input = Array1::from_vec(vec![1.0, 2.0]);
        let mut output = Array1::zeros(3);
        
        // Forward pass
        layer.forward(&input, &mut output);
        
        // Expected output:
        // [1.0, 2.0] * [1.0, 2.0; 3.0, 4.0; 5.0, 6.0] + [0.1, 0.2, 0.3]
        // = [5.1, 11.2, 17.3]
        assert!((output[0] - 5.1).abs() < 1e-6);
        assert!((output[1] - 11.2).abs() < 1e-6);
        assert!((output[2] - 17.3).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        // Create test input
        let mut input = Array1::from_vec(vec![-1.0, 0.0, 2.0, -3.5, 4.2]);
        
        // Create ReLU layer
        let relu = ReLULayer::new(5);
        
        // In-place activation
        relu.activate(&mut input);
        
        // ReLU should zero out negative values and pass through positives
        assert!((input[0] - 0.0).abs() < 1e-6);    // -1.0 -> 0.0
        assert!((input[1] - 0.0).abs() < 1e-6);    // 0.0 -> 0.0 
        assert!((input[2] - 2.0).abs() < 1e-6);    // 2.0 -> 2.0
        assert!((input[3] - 0.0).abs() < 1e-6);    // -3.5 -> 0.0
        assert!((input[4] - 4.2).abs() < 1e-6);    // 4.2 -> 4.2
    }

    #[test]
    fn test_softmax() {
        // Create test input
        let mut input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        
        // Create Softmax layer
        let softmax = SoftmaxLayer::new(4);
        
        // In-place activation
        softmax.activate(&mut input);
        
        // Check that all values are between 0 and 1
        for &x in input.iter() {
            assert!(x >= 0.0 && x <= 1.0);
        }
        
        // Check that sum is 1
        let sum: f32 = input.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that larger inputs give larger probabilities
        assert!(input[3] > input[2]);
        assert!(input[2] > input[1]);
        assert!(input[1] > input[0]);
    }
}