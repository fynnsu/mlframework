use std::ops::{Add, Div, Mul, Sub};

use crate::{dtype::Dtype, ops::Op, tensor::Tensor};

use super::vec::zeros_like;

// Ops
impl<T: Dtype> Add for Tensor<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Op::ElAdd(self, other).forward()
    }
}

impl<T: Dtype> Sub for Tensor<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Op::ElSub(self, other).forward()
    }
}

impl<T: Dtype> Mul for Tensor<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Op::ElMul(self, other).forward()
    }
}

impl<T: Dtype> Div for Tensor<T> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Op::ElDiv(self, other).forward()
    }
}

impl<T: Dtype> Tensor<T> {
    pub fn reduce_sum(self) -> Self {
        Op::ReduceSum(self).forward()
    }

    pub fn max(self, other: Self) -> Self {
        Op::ElMax(self, other).forward()
    }

    pub fn min(self, other: Self) -> Self {
        Op::ElMin(self, other).forward()
    }

    pub fn relu(self) -> Self {
        let other = Tensor::new(zeros_like(self.data.as_ref()));
        Op::ElMax(self, other).forward()
    }
}

// Backwards Fns
