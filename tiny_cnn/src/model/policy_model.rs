use ndarray::{Array4, Array2, Array3, Array1};
use crate::layers::{Layer, Conv2dLayer, PaddedConv2dLayer, ReLULayer, DenseLayer, SoftmaxLayer};

pub struct PolicyModel {
    // Network architecture constants
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    num_actions: usize,
    conv1_channels: usize,
    conv2_channels: usize,
    conv2_padding: usize,
    fc1_channels: usize,

    // Layers
    conv1: PaddedConv2dLayer,
    relu1: ReLULayer,
    conv2: Conv2dLayer,
    relu2: ReLULayer,
    fc1: DenseLayer,
    relu3: ReLULayer,
    fc2: DenseLayer,
    softmax: SoftmaxLayer,

    // Input/output buffers
    input: Array3<f32>,  // (1, 4, 4) for the game board
    conv1_output: Array3<f32>,  // (4, 4, 4)
    conv2_output: Array3<f32>,  // (16, 2, 2)
    fc1_input: Array1<f32>,     // (64) flattened conv2 output
    fc1_output: Array1<f32>,    // (32)
    fc2_output: Array1<f32>,    // (4)
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
        // Network architecture constants
        let input_channels = 1;
        let input_height = 4;
        let input_width = 4;
        let num_actions = 4;
        let conv1_channels = 4;
        let conv2_channels = 16;
        let conv2_padding = 0;
        let fc1_channels = 32;

        // Create layers
        let conv1 = PaddedConv2dLayer::new(
            conv1_channels, input_channels, input_height, input_width, 3, 
            conv1_weight, conv1_bias
        );
        let relu1 = ReLULayer::new(conv1_channels * input_height * input_width);
        
        let conv2 = Conv2dLayer::new(
            conv2_channels, conv1_channels, input_height, input_width, 3, 
            conv2_weight, conv2_bias
        );
        let relu2 = ReLULayer::new(conv2_channels * (input_height - 2) * (input_width - 2));
        
        let fc1 = DenseLayer::new(
            fc1_channels, 
            conv2_channels * (input_height - 2) * (input_width - 2),
            fc1_weight, 
            fc1_bias
        );
        let relu3 = ReLULayer::new(fc1_channels);
        
        let fc2 = DenseLayer::new(
            num_actions, 
            fc1_channels,
            fc2_weight, 
            fc2_bias
        );
        let softmax = SoftmaxLayer::new(num_actions);

        // Initialize buffers
        let input = Array3::zeros((input_channels, input_height, input_width));
        let conv1_output = Array3::zeros((conv1_channels, input_height, input_width));
        let conv2_output = Array3::zeros((conv2_channels, input_height - 2, input_width - 2));
        let fc1_input = Array1::zeros(conv2_channels * (input_height - 2) * (input_width - 2));
        let fc1_output = Array1::zeros(fc1_channels);
        let fc2_output = Array1::zeros(num_actions);

        Self {
            input_channels,
            input_height,
            input_width,
            num_actions,
            conv1_channels,
            conv2_channels,
            conv2_padding,
            fc1_channels,
            conv1, relu1, conv2, relu2, fc1, relu3, fc2, softmax,
            input, conv1_output, conv2_output, fc1_input, fc1_output, fc2_output,
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
        self.softmax.activate(&mut self.fc2_output);

        &self.fc2_output
    }
}
