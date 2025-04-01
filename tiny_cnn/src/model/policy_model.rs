use ndarray::{Array4, Array2, Array3, Array1};
use crate::layers::{Layer, Conv2dLayer, PaddedConv2dLayer, ReLULayer, DenseLayer, SoftmaxLayer};

pub struct PolicyModel {
    // Layers
    conv1: PaddedConv2dLayer,
    relu1: ReLULayer,
    conv2: PaddedConv2dLayer,
    relu2: ReLULayer,
    fc1: DenseLayer,
    relu3: ReLULayer,
    fc2: DenseLayer,
    softmax: SoftmaxLayer,

    // Input/output buffers
    input: Array3<f32>,  // (1, 4, 4) for the game board
    conv1_output: Array3<f32>,  // (32, 4, 4)
    conv2_output: Array3<f32>,  // (64, 4, 4)
    fc1_input: Array1<f32>,     // (1024) flattened conv2 output
    fc1_output: Array1<f32>,    // (256)
    fc2_output: Array1<f32>,    // (128)
    output: Array1<f32>,        // (4) for the action probabilities
}

impl PolicyModel {
    pub fn new(
        conv1_weight: Array4<f32>,
        conv1_bias: Array1<f32>,
        conv2_weight: Array4<f32>,
        conv2_bias: Array1<f32>,
        fc1_weight: Array2<f32>,
        fc1_bias: Array1<f32>,
        fc2_weight: Array2<f32>,
        fc2_bias: Array1<f32>,
    ) -> Self {
        // Create layers
        let conv1 = PaddedConv2dLayer::new(32, 1, 4, 4, 3, conv1_weight, conv1_bias);
        let relu1 = ReLULayer::new(32 * 4 * 4);
        let conv2 = PaddedConv2dLayer::new(64, 32, 4, 4, 3, conv2_weight, conv2_bias);
        let relu2 = ReLULayer::new(64 * 4 * 4);
        let fc1 = DenseLayer::new(256, 1024, fc1_weight, fc1_bias);
        let relu3 = ReLULayer::new(256);
        let fc2 = DenseLayer::new(128, 256, fc2_weight, fc2_bias);
        let softmax = SoftmaxLayer::new(4);

        // Initialize buffers
        let input = Array3::zeros((1, 4, 4));
        let conv1_output = Array3::zeros((32, 4, 4));
        let conv2_output = Array3::zeros((64, 4, 4));
        let fc1_input = Array1::zeros(1024);
        let fc1_output = Array1::zeros(256);
        let fc2_output = Array1::zeros(128);
        let output = Array1::zeros(4);

        Self {
            conv1, relu1, conv2, relu2, fc1, relu3, fc2, softmax,
            input, conv1_output, conv2_output, fc1_input, fc1_output, fc2_output, output,
        }
    }

    pub fn inference(&mut self, board: &Array3<f32>) -> &Array1<f32> {
        // Copy input board to our buffer
        self.input.assign(board);

        // Forward pass through the network
        self.conv1.forward(&self.input, &mut self.conv1_output);
        
        // Reshape conv1_output to 1D and apply ReLU
        self.fc1_input.assign(&Array1::from_vec(self.conv1_output.iter().copied().collect()));
        self.relu1.activate(&mut self.fc1_input);
        
        self.conv2.forward(&self.conv1_output, &mut self.conv2_output);
        
        // Reshape conv2_output to 1D and apply ReLU
        self.fc1_input.assign(&Array1::from_vec(self.conv2_output.iter().copied().collect()));
        self.relu2.activate(&mut self.fc1_input);
        
        self.fc1.forward(&self.fc1_input, &mut self.fc1_output);
        self.relu3.activate(&mut self.fc1_output);
        
        self.fc2.forward(&self.fc1_output, &mut self.fc2_output);
        self.relu3.activate(&mut self.fc2_output);
        
        // Final dense layer and softmax
        self.fc2.forward(&self.fc2_output, &mut self.output);
        self.softmax.activate(&mut self.output);

        &self.output
    }
}
