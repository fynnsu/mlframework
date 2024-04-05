# mlframework

Building a rudimentary ml framework from scratch in rust.

> [!NOTE]
> This project is a work in progess. It is also intended as a learning project to improve my understanding of ML framework implementation details and Rust.

## Features

### Basic operations with backprop

```rust
let x = Tensor::new([3., 4., 5.]);
let y = Tensor::new([1., -2., 1.]);
let z = Tensor::new([-3., 1., 3.]);
let s = ((x + y.clone() + y) * z.relu()).reduce_sum();

// Performs reverse mode autodiff and sets private grad field on tensors throughout the graph of `s`.
s.backward();
```

### Compile-time shape checking

```rust
let x = Tensor::new([[0.0; 5]; 10]); // shape: [10, 5]

// Simply specify a Tensory w/ dtype and shape and call reshape
// If the shape is valid for the starting shape the code will compile
let x2: Tensor<f32, D3<2, 5, 5>> = x.reshape(); // shape: [2, 5, 5]

// This will *not* compile, becase the shape [51] is invalid for a tensor of shape [2, 5, 5]
let x3: Tensor<f32, D1<51>> = x2.reshape(); // ERROR!!
```

## Status

- [x] Initial tensor structure
- [x] Basic elementwise operations
- [x] Basic backprop implementation
  - [x] Fix graph traversal complexity
- [x] Tensor Shapes
- [x] Matmul
- [ ] Tensor Indexing
- [x] Potentially improve Op implementation
  - [x] Make it simpler to create/register new ops & backward functions
  - [ ] Generate similar ops using macros
- [ ] Define a module structure
- [ ] Implement an optimizer
- [ ] Setup test structure
- [ ] Organize repo code
- [x] Add CI w/ Github Actions
