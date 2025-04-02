# 2048 Game Implementation

This document describes the data transformations used in the 2048 game implementation.

## Move Direction Encoding

The game uses a `MoveDirection` enum to represent the four possible moves:

```rust
pub enum MoveDirection {
    Up,
    Right,
    Down,
    Left,
}
```

### Direction to Number Mapping

When converting directions to numbers (e.g., for neural network output), we use the following mapping:

| Direction | Index |
|-----------|-------|
| Up        | 0     |
| Right     | 1     |
| Down      | 2     |
| Left      | 3     |

This mapping is consistent across:
- Neural network output (4 probabilities)
- Policy representation
- Move selection

### Direction Iteration

The `MoveDirection` enum provides two ways to iterate over all directions:
1. `MoveDirection::all()` - Returns an array of all directions
2. `MoveDirection::iter()` - Returns an iterator over all directions

The order is always: Up, Right, Down, Left

## Board Position Encoding

The game board is a 4x4 grid that can be represented in different formats:

### Position to Array (16 values)

When converting a board position to a neural network input, we use the following transformation:

```rust
fn board_to_input(&self, position: &Position) -> [f32; 16] {
    let mut input = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            if let Some(tile) = position.get_tile(i, j) {
                input[i * 4 + j] = tile.value() as f32;
            }
        }
    }
    input
}
```

The array is flattened in row-major order:
```
[0,0] [0,1] [0,2] [0,3]
[1,0] [1,1] [1,2] [1,3]
[2,0] [2,1] [2,2] [2,3]
[3,0] [3,1] [3,2] [3,3]
```

### Neural Network Input Format

The neural network expects input in the following format:
- Shape: [1, 4, 4] (batch_size=1, height=4, width=4)
- Values: Raw tile values (0 for empty, 2, 4, 8, etc. for tiles)
- The input is then preprocessed in the network:
  - Take log2 of non-zero values
  - Keep zero values as zero

### Neural Network Output Format

The neural network outputs:
- Shape: [4] (one probability per direction)
- Values: Probabilities for each direction (sum to 1)
- Order: [Up, Right, Down, Left]

## Consistency Notes

1. Direction Order:
   - Always use `MoveDirection::all()` or `MoveDirection::iter()` to ensure consistent ordering
   - Never assume a specific order when iterating over directions

2. Board Representation:
   - Use row-major order for flattening (i * 4 + j)
   - Keep empty tiles as 0
   - Use raw values for input, let the network handle preprocessing

3. Policy Representation:
   - Always use the same direction order: Up, Right, Down, Left
   - Ensure probabilities sum to 1
   - Use the same indices for directions in all policy-related code