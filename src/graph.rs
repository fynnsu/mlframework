use std::collections::HashMap;

use crate::{dtype::Dtype, tensor::Tensor};

#[derive(Debug)]
pub struct TensorGraph<'a, T: Dtype> {
    pub nodes: HashMap<usize, &'a Tensor<T>>,
    pub edges: HashMap<usize, Vec<usize>>,
}

impl<'a, T: Dtype> TensorGraph<'a, T> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }
}
