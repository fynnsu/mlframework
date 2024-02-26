use crate::{dtype::Dtype, tensor::Tensor};

#[derive(Debug, Clone)]
pub enum Op<T: Dtype> {
    ElementwisePair(ElementwiseBinary, Tensor<T>, Tensor<T>),
}

#[derive(Debug, Copy, Clone)]
pub enum ElementwiseBinary {
    Add,
    Sub,
    Mul,
    Div,
}

impl ElementwiseBinary {
    pub fn _f<T: Dtype>(self, a: &T, b: &T) -> T {
        match self {
            ElementwiseBinary::Add => *a + *b,
            ElementwiseBinary::Sub => *a - *b,
            ElementwiseBinary::Mul => *a * *b,
            ElementwiseBinary::Div => *a / *b,
        }
    }
}
