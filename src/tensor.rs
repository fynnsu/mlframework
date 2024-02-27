use std::{
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use crate::op::{ElementwiseBinary, Op};
use crate::{dtype::Dtype, op::ElementwiseUnary};

#[derive(Debug)]
pub struct Tensor<T: Dtype> {
    pub _data: Rc<Vec<T>>,
    pub _op: Option<Box<Op<T>>>,
}

impl<T: Dtype> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            _data: Rc::new(data),
            _op: None,
        }
    }

    fn elementwise_binary_op(self, eb: ElementwiseBinary, other: Self) -> Self {
        assert!(self._data.len() == other._data.len());
        Self {
            _data: Rc::new(
                self._data
                    .iter()
                    .zip(other._data.iter())
                    .map(|(a, b)| eb._f(a, b))
                    .collect(),
            ),
            _op: Some(Box::new(Op::EB(eb, self, other))),
        }
    }

    fn elementwise_unary_op(self, eu: ElementwiseUnary) -> Self {
        Self {
            _data: Rc::new(self._data.iter().map(|a| eu._f(a)).collect()),
            _op: Some(Box::new(Op::EU(eu, self))),
        }
    }

    pub fn relu(self) -> Self {
        self.elementwise_unary_op(ElementwiseUnary::Relu)
    }
}

impl<T: Dtype> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            _data: Rc::clone(&self._data),
            _op: match &self._op {
                Some(b) => Some(Box::new(b.as_ref().clone())),
                None => None,
            },
        }
    }
}

impl<T: Dtype> Add for Tensor<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.elementwise_binary_op(ElementwiseBinary::Add, other)
    }
}
impl<T: Dtype> Sub for Tensor<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.elementwise_binary_op(ElementwiseBinary::Sub, other)
    }
}
impl<T: Dtype> Mul for Tensor<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.elementwise_binary_op(ElementwiseBinary::Mul, other)
    }
}
impl<T: Dtype> Div for Tensor<T> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.elementwise_binary_op(ElementwiseBinary::Div, other)
    }
}
