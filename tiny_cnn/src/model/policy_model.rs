use ndarray::{Array4, Array2, ArrayView4, ArrayView2};
pub struct PolicyModel {
    // Model parameters
    conv1_weight: Array4<f32>,
    conv1_bias: Array2<f32>,
    conv2_weight: Array4<f32>,
    conv2_bias: Array2<f32>,
    fc1_weight: Array2<f32>,
    fc1_bias: Array2<f32>,
    fc2_weight: Array2<f32>,
    fc2_bias: Array2<f32>,
}

impl PolicyModel {
    pub fn new() -> Self {
        // Initialize with zeros for now, will be loaded from npz
        Self {
            conv1_weight: Array4::zeros((4, 1, 3, 3)),  // (out_channels, in_channels, kernel_size, kernel_size)
            conv1_bias: Array2::zeros((4, 1)),
            conv2_weight: Array4::zeros((16, 4, 3, 3)),
            conv2_bias: Array2::zeros((16, 1)),
            fc1_weight: Array2::zeros((32, 64)),  // 64 = 16 * 2 * 2 (channels * height * width)
            fc1_bias: Array2::zeros((32, 1)),
            fc2_weight: Array2::zeros((4, 32)),
            fc2_bias: Array2::zeros((4, 1)),
        }
    }

    pub fn from_arrays(
        conv1_weight: Array4<f32>,
        conv1_bias: Array2<f32>,
        conv2_weight: Array4<f32>,
        conv2_bias: Array2<f32>,
        fc1_weight: Array2<f32>,
        fc1_bias: Array2<f32>,
        fc2_weight: Array2<f32>,
        fc2_bias: Array2<f32>,
    ) -> Self {
        Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        }
    }
    pub fn forward(&self, x: ArrayView4<f32>) -> Array2<f32> {
        // First conv block
        let x = self.conv2d(x, &self.conv1_weight, &self.conv1_bias);
        let x = self.relu_4d(x);

        // Second conv block
        let x = self.conv2d(x.view(), &self.conv2_weight, &self.conv2_bias);
        let x = self.relu_4d(x);

        // Flatten
        let x = x.into_shape([10, 10]).unwrap();//TODO

        // Fully connected layers
        let x = self.linear(x, &self.fc1_weight, &self.fc1_bias);
        let x = self.relu_2d(x);
        //let x = self.dropout(x, 0.5);
        let x = self.linear(x, &self.fc2_weight, &self.fc2_bias);

        // Softmax
        self.softmax(x)
    }

    fn conv2d(&self, x: ArrayView4<f32>, weight: &Array4<f32>, bias: &Array2<f32>) -> Array4<f32> {
        // Simple 2D convolution implementation
        let (batch_size, in_channels, height, width) = x.dim();
        let (out_channels, _, kernel_size, _) = weight.dim();
        let output_height = height - kernel_size + 1;
        let output_width = width - kernel_size + 1;
        
        let mut output = Array4::zeros((batch_size, out_channels, output_height, output_width));
        
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let mut sum = 0.0;
                        for ic in 0..in_channels {
                            for kh in 0..kernel_size {
                                for kw in 0..kernel_size {
                                    sum += x[[b, ic, h + kh, w + kw]] * weight[[oc, ic, kh, kw]];
                                }
                            }
                        }
                        output[[b, oc, h, w]] = sum + bias[[oc, 0]];
                    }
                }
            }
        }
        output
    }

    fn linear(&self, x: Array2<f32>, weight: &Array2<f32>, bias: &Array2<f32>) -> Array2<f32> {
        x.dot(weight) + bias.t()
    }

    fn relu_4d(&self, x: Array4<f32>) -> Array4<f32> {
        x.map(|&v| v.max(0.0))
    }

    fn relu_2d(&self, x: Array2<f32>) -> Array2<f32> {
        x.map(|&v| v.max(0.0))
    }

    fn dropout(&self, x: Array2<f32>, p: f32) -> Array2<f32> {
        let mask = Array2::from_shape_fn(x.dim(), |_| if rand::random::<f32>() > p { 1.0 } else { 0.0 });
        x * mask / (1.0 - p)
    }

    fn softmax(&self, x: Array2<f32>) -> Array2<f32> {
        let exp = x.map(|&v| v.exp());
        let sum = exp.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
        exp / sum
    }
} 