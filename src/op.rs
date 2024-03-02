use std::collections::HashMap;

use crate::{dtype::Dtype, tensor::Tensor};

#[derive(Debug, Clone)]
pub enum Op<T: Dtype> {
    EB(ElementwiseBinary, Tensor<T>, Tensor<T>),
    EU(ElementwiseUnary, Tensor<T>),
    Sum(Tensor<T>),
}

#[derive(Debug, Copy, Clone)]
pub enum ElementwiseUnary {
    Relu,
}

fn ones_like<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    vec![T::one(); a.len()]
}

fn expand_to_shape<T: Dtype>(a: &Vec<T>, len: usize) -> Vec<T> {
    assert_eq!(a.len(), 1);
    vec![a[0]; len]
}

fn el_bin<T: Dtype, F>(op: F, a: &Vec<T>, b: &Vec<T>) -> Vec<T>
where
    F: Fn((&T, &T)) -> T,
{
    a.iter().zip(b.iter()).map(op).collect()
}

fn el_mul<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| *x * *y, a, b)
}

fn el_add<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| *x + *y, a, b)
}

fn el_div<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| *x / *y, a, b)
}

fn el_unary<T: Dtype, F>(op: F, a: &Vec<T>) -> Vec<T>
where
    F: Fn(&T) -> T,
{
    a.iter().map(op).collect()
}

fn el_neg<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    el_unary(|x| T::neg(*x), a)
}

fn el_relu<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    el_unary(|x| if *x >= T::zero() { *x } else { T::zero() }, a)
}

fn el_inv<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    el_unary(|x| T::one() / *x, a)
}

fn el_pos<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    // Used for relu grad
    // Return 1 if x >= 0, else 0
    el_unary(|x| if *x >= T::zero() { T::one() } else { T::zero() }, a)
}

impl<T: Dtype> Op<T> {
    pub fn iter(&self) -> impl Iterator<Item = &Tensor<T>> {
        match self {
            Op::EB(_, a, b) => vec![a, b].into_iter(),
            Op::EU(_, a) => vec![a].into_iter(),
            Op::Sum(a) => vec![a].into_iter(),
        }
    }

    pub fn propogate_grad(
        &self,
        mut grads: HashMap<usize, Vec<T>>,
        p_id: usize,
    ) -> HashMap<usize, Vec<T>> {
        let d_dt = grads.get(&p_id).unwrap();
        match self {
            Op::EB(eb, a, b) => {
                let (d_da, d_db) = match eb {
                    ElementwiseBinary::Add => {
                        let dt_da = ones_like(a.data.as_ref());
                        let dt_db = ones_like(b.data.as_ref());
                        (el_mul(d_dt, &dt_da), el_mul(d_dt, &dt_db))
                    }
                    ElementwiseBinary::Sub => {
                        let dt_da = ones_like(a.data.as_ref());
                        let dt_db = el_neg(&ones_like(b.data.as_ref()));
                        (el_mul(d_dt, &dt_da), el_mul(d_dt, &dt_db))
                    }
                    ElementwiseBinary::Mul => {
                        let dt_da = b.data.as_ref();
                        let dt_db = a.data.as_ref();
                        (el_mul(d_dt, dt_da), el_mul(d_dt, dt_db))
                    }
                    ElementwiseBinary::Div => {
                        // t = a / b = ab^(-1)
                        // dt/db = -a * b^(-2)
                        let dt_da = el_inv(b.data.as_ref());
                        let dt_db = el_mul(
                            &el_neg(a.data.as_ref()),
                            &el_inv(&el_mul(b.data.as_ref(), b.data.as_ref())),
                        );
                        (el_mul(d_dt, &dt_da), el_mul(d_dt, &dt_db))
                    }
                };
                grads
                    .entry(a.id)
                    .and_modify(|v| *v = el_add(v, &d_da))
                    .or_insert(d_da);
                grads
                    .entry(b.id)
                    .and_modify(|v| *v = el_add(v, &d_db))
                    .or_insert(d_db);
            }
            Op::EU(eu, a) => {
                let d_da = match eu {
                    ElementwiseUnary::Relu => {
                        let dt_da = el_pos(a.data.as_ref());
                        el_mul(d_dt, &dt_da)
                    }
                };
                grads
                    .entry(a.id)
                    .and_modify(|v| *v = el_add(v, &d_da))
                    .or_insert(d_da);
            }
            Op::Sum(a) => {
                let dt_da = ones_like(a.data.as_ref());
                let d_dt_expanded = expand_to_shape(d_dt, a.data.len());
                let d_da = el_mul(&d_dt_expanded, &dt_da);
                grads
                    .entry(a.id)
                    .and_modify(|v| *v = el_add(v, &d_da))
                    .or_insert(d_da);
            }
        }
        grads
    }
}

impl ElementwiseUnary {
    pub fn _f<T: Dtype>(self, a: &T) -> T {
        match self {
            Self::Relu => {
                if *a >= T::zero() {
                    *a
                } else {
                    T::zero()
                }
            }
        }
    }
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
