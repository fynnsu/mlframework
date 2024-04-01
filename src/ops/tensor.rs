use std::{
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use crate::{
    dtype::Dtype,
    ops::Op,
    shape::{Const, Shape},
    tensor::Tensor,
};

use crate::ops::vec::{el_add, el_div, el_max, el_min, el_mul, el_sub};

use super::vec::el_relu;

macro_rules! impl_bin_el_op {
    ($s:ident, $t:ident, $tf:ident, $f:expr) => {
        impl<T: Dtype, S: Shape> Op for $s<T, S> {
            type Produces = Tensor<T, S>;

            fn propogate_grad(&self, t: &Self::Produces) {
                todo!()
            }
            // let data = $f(&self.0.data, &self.1.data).into();

            fn forward(self) -> Self::Produces {
                let data = $f(&self.0.data, &self.1.data).into();
                unsafe { Self::Produces::from_rc_td_and_op_unchecked(Rc::new(data), Rc::new(self)) }
            }

            fn clone(&self) -> Self
            where
                Self: Sized,
            {
                todo!()
            }
        }

        impl<T: Dtype, S: Shape> $t for Tensor<T, S> {
            type Output = Self;
            fn $tf(self, other: Self) -> Self {
                $s(self, other).forward()
            }
        }
    };
}

// Ops

#[derive(Debug)]
pub struct ElAddStruct<T: Dtype, S: Shape>(Tensor<T, S>, Tensor<T, S>);

#[derive(Debug)]
pub struct ElSubStruct<T: Dtype, S: Shape>(Tensor<T, S>, Tensor<T, S>);

#[derive(Debug)]
pub struct ElMulStruct<T: Dtype, S: Shape>(Tensor<T, S>, Tensor<T, S>);

#[derive(Debug)]
pub struct ElDivStruct<T: Dtype, S: Shape>(Tensor<T, S>, Tensor<T, S>);

#[derive(Debug)]
pub struct ElMaxStruct<T: Dtype, S: Shape>(Tensor<T, S>, Tensor<T, S>);

#[derive(Debug)]
pub struct ElMinStruct<T: Dtype, S: Shape>(Tensor<T, S>, Tensor<T, S>);

#[derive(Debug)]
pub struct ElReLUStruct<T: Dtype, S: Shape>(Tensor<T, S>);

#[derive(Debug)]
pub struct ReduceSumStruct<T: Dtype, S: Shape>(Tensor<T, S>);

impl_bin_el_op!(ElAddStruct, Add, add, el_add);
impl_bin_el_op!(ElSubStruct, Sub, sub, el_sub);
impl_bin_el_op!(ElMulStruct, Mul, mul, el_mul);
impl_bin_el_op!(ElDivStruct, Div, div, el_div);
impl_bin_el_op!(ElMaxStruct, Max, max, el_max);
impl_bin_el_op!(ElMinStruct, Min, min, el_min);

pub trait Max<Rhs = Self> {
    type Output;

    fn max(self, other: Self) -> Self;
}

pub trait Min<Rhs = Self> {
    type Output;

    fn min(self, other: Self) -> Self;
}

// ReLU
impl<T: Dtype, S: Shape> Op for ElReLUStruct<T, S> {
    type Produces = Tensor<T, S>;

    fn propogate_grad(&self, t: &Self::Produces) {
        todo!()
    }
    // let data = $f(&self.0.data, &self.1.data).into();

    fn forward(self) -> Self::Produces {
        let data = el_relu(&self.0.data).into();
        unsafe { Self::Produces::from_rc_td_and_op_unchecked(Rc::new(data), Rc::new(self)) }
    }

    fn clone(&self) -> Self
    where
        Self: Sized,
    {
        todo!()
    }
}

impl<T: Dtype, S: Shape> Tensor<T, S> {
    pub fn relu(self) -> Self {
        ElReLUStruct(self).forward()
    }

    pub fn reduce_sum(self) -> Tensor<T, (Const<1>,)> {
        ReduceSumStruct(self).forward()
    }
}

// Reduce sum
impl<T: Dtype, S: Shape> Op for ReduceSumStruct<T, S> {
    type Produces = Tensor<T, (Const<1>,)>;

    fn propogate_grad(&self, t: &Self::Produces) {
        todo!()
    }
    // let data = $f(&self.0.data, &self.1.data).into();

    fn forward(self) -> Self::Produces {
        let data = vec![(self.0.data.iter().fold(T::zero(), |s, x| s + *x))].into();
        unsafe { Self::Produces::from_rc_td_and_op_unchecked(Rc::new(data), Rc::new(self)) }
    }

    fn clone(&self) -> Self
    where
        Self: Sized,
    {
        todo!()
    }
}
