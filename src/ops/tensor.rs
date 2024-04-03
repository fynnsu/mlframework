use crate::ops::grad::{
    el_add_grad, el_div_grad, el_max_grad, el_min_grad, el_mul_grad, el_relu_grad, el_sub_grad,
};
use crate::ops::vec::{el_add, el_div, el_max, el_min, el_mul, el_relu, el_sub};
use crate::tensor::TensorBox;
use crate::{
    dtype::Dtype,
    ops::Op,
    shape::{Const, Shape},
    tensor::Tensor,
};
use std::{
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use super::grad::reduce_sum_grad;
use super::vec::expand_to_shape;

macro_rules! impl_bin_el_op {
    ($s:ident, $t:ident, $tf:ident, $f:expr, $df:expr) => {
        impl<T: Dtype, S: Shape> Op for $s<T, S> {
            type Produces = Tensor<T, S>;

            fn propogate_grad(&self, t: &Self::Produces) {
                // t = f(a, b)
                if let Some(d_dt) = t.data.grad_ref().as_ref() {
                    let (dt_da, dt_db) = $df(self.0.data.as_ref(), self.1.data.as_ref());
                    let d_da = el_mul(d_dt, &dt_da);
                    let d_db = el_mul(d_dt, &dt_db);
                    self.0.data.update_grad(d_da);
                    self.1.data.update_grad(d_db);
                } else {
                    panic!("Attempted to propogate grad, but no grad value exists.")
                }
            }
            // let data = $f(&self.0.data, &self.1.data).into();

            fn forward(self) -> Self::Produces {
                let data = $f(&self.0.data, &self.1.data).into();
                unsafe { Self::Produces::from_rc_td_and_op_unchecked(Rc::new(data), Rc::new(self)) }
            }

            fn operands(&self) -> Vec<TensorBox> {
                vec![
                    TensorBox::new(self.0.id, &self.0),
                    TensorBox::new(self.1.id, &self.1),
                ]
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

impl_bin_el_op!(ElAddStruct, Add, add, el_add, el_add_grad);
impl_bin_el_op!(ElSubStruct, Sub, sub, el_sub, el_sub_grad);
impl_bin_el_op!(ElMulStruct, Mul, mul, el_mul, el_mul_grad);
impl_bin_el_op!(ElDivStruct, Div, div, el_div, el_div_grad);
impl_bin_el_op!(ElMaxStruct, Max, max, el_max, el_max_grad);
impl_bin_el_op!(ElMinStruct, Min, min, el_min, el_min_grad);

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
        // t = max(a, 0)
        // dt_da = 1 if a > 0 else 0
        if let Some(d_dt) = t.data.grad_ref().as_ref() {
            let dt_da = el_relu_grad(self.0.data.as_ref());
            let d_da = el_mul(d_dt, &dt_da);
            self.0.data.update_grad(d_da);
        } else {
            panic!("Attempted to propogate grad, but no grad value exists.")
        }
    }

    fn forward(self) -> Self::Produces {
        let data = el_relu(&self.0.data).into();
        unsafe { Self::Produces::from_rc_td_and_op_unchecked(Rc::new(data), Rc::new(self)) }
    }

    fn operands(&self) -> Vec<TensorBox> {
        vec![TensorBox::new(self.0.id, &self.0)]
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
        // t = reduce_sum(a)
        if let Some(d_dt) = t.data.grad_ref().as_ref() {
            let dt_da = reduce_sum_grad(self.0.data.as_ref());
            let d_dt_expanded = expand_to_shape(d_dt, dt_da.len());
            let d_da = el_mul(&d_dt_expanded, &dt_da);
            self.0.data.update_grad(d_da);
        } else {
            panic!("Attempted to propogate grad, but no grad value exists.")
        }
    }

    fn forward(self) -> Self::Produces {
        let data = vec![(self.0.data.iter().fold(T::zero(), |s, x| s + *x))].into();
        unsafe { Self::Produces::from_rc_td_and_op_unchecked(Rc::new(data), Rc::new(self)) }
    }

    fn operands(&self) -> Vec<TensorBox> {
        vec![TensorBox::new(self.0.id, &self.0)]
    }
}
