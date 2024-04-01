use std::{
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use crate::{
    dtype::Dtype,
    ops::Op,
    shape::{Const, Shape},
    tensor::Tensor,
    tensor_data::TensorData,
};

use crate::ops::vec::{el_add, el_div, el_max, el_min, el_mul, el_sub};

use super::vec::zeros_like;

// Ops

#[derive(Debug)]
pub struct ElAddStruct<T: Dtype, S: Shape>(Tensor<T, S>, Tensor<T, S>);

impl<T: Dtype, S: Shape> Op for ElAddStruct<T, S> {
    type Produces = Tensor<T, S>;

    fn propogate_grad(&self, t: &Self::Produces) {
        todo!()
    }

    fn forward(self) -> Self::Produces {
        let data = el_add(&self.0.data, &self.1.data).into();
        unsafe { Self::Produces::from_rc_td_and_op_unchecked(Rc::new(data), Rc::new(self)) }
    }

    fn clone(&self) -> Self
    where
        Self: Sized,
    {
        todo!()
    }
}

impl<T: Dtype, S: Shape> Add for Tensor<T, S> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        ElAddStruct(self, other).forward()
    }
}

#[derive(Debug)]
pub struct ElSubStruct<T: Dtype, S: Shape>(Tensor<T, S>, Tensor<T, S>);

impl<T: Dtype, S: Shape> Op for ElSubStruct<T, S> {
    type Produces = Tensor<T, S>;

    fn propogate_grad(&self, t: &Self::Produces) {
        todo!()
    }

    fn forward(self) -> Self::Produces {
        let data = el_sub(&self.0.data, &self.1.data).into();
        unsafe { Self::Produces::from_rc_td_and_op_unchecked(Rc::new(data), Rc::new(self)) }
    }

    fn clone(&self) -> Self
    where
        Self: Sized,
    {
        todo!()
    }
}

impl<T: Dtype, S: Shape> Sub for Tensor<T, S> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        ElSubStruct(self, other).forward()
    }
}

// impl<T: Dtype, S: Shape> Mul for Tensor<T, S> {
//     type Output = Self;
//     fn mul(self, other: Self) -> Self {
//         Op::ElMul(self, other).forward()
//     }
// }

// impl<T: Dtype, S: Shape> Div for Tensor<T, S> {
//     type Output = Self;
//     fn div(self, other: Self) -> Self {
//         Op::ElDiv(self, other).forward()
//     }
// }

// impl<T: Dtype, S: Shape> Tensor<T, S> {
//     pub fn reduce_sum(self) -> Tensor<T, (Const<1>,)> {
//         Op::ReduceSum(self).forward()
//     }

//     pub fn max(self, other: Self) -> Self {
//         Op::ElMax(self, other).forward()
//     }

//     pub fn min(self, other: Self) -> Self {
//         Op::ElMin(self, other).forward()
//     }

//     pub fn relu(self) -> Self {
//         let other = Tensor::new(zeros_like(self.data.as_ref()));
//         Op::ElMax(self, other).forward()
//     }
// }

// Backwards Fns
