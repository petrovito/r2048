use ndarray::Array3;
use crate::model::policy_model::PolicyModel;

/// Runs inference on a flattened board state (16 values)
/// Returns a probability distribution over 4 moves
pub fn run_inference(board: &[f32; 16], model: &mut PolicyModel) -> [f32; 4] {
    // Reshape the input into a 4x4x1 array and preprocess
    let mut input = Array3::zeros((1, 4, 4));
    for i in 0..4 {
        for j in 0..4 {
            let value = board[i * 4 + j];
            // Only take log2 of nonzero values
            input[[0, i, j]] = if value > 0.0 {
                value.log2()
            } else {
                0.0
            };
        }
    }

    // Run inference
    let output = model.inference(&input);

    // Convert output to array
    let mut result = [0.0; 4];
    for i in 0..4 {
        result[i] = output[i];
    }
    result
} 