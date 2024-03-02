# mlframework

Building a rudimentary ml framework from scratch in rust.

> [!NOTE]  
> This project is a work in progess. It is also intended as a learning project to improve my understanding of ML framework implementation details and Rust.

## Status
``` rust
let x = Tensor::new(vec![3., 4., 5.]);
let y = Tensor::new(vec![1., -2., 1.]);
let z = Tensor::new(vec![-3., 1., 3.]);
let s = ((x + y.clone() + y) * z.relu()).sum();

// produces a hashmap mapping tensor_id: usize -> gradient: Vec<T>
let grad = s.grad();
```

- [x] Initial tensor structure
- [x] Basic elementwise operations
- [x] Basic backprop implementation
    - [ ] Fix graph traversal complexity
- [ ] Tensor Shapes
- [ ] Matmul
- [ ] Tensor Indexing
- [ ] Potentially improve Op implementation
    - [ ] Make it simpler to create/register new ops & backward functions
    - [ ] Generate similar ops using macros
- [ ] Setup test structure
- [ ] Organize repo code
