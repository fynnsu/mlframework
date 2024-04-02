use super::vec::{el_bin, el_gt, el_inv, el_lt, el_neg, el_pos, ones_like};
use crate::dtype::Dtype;
use std::borrow::Cow;

pub(crate) fn el_add_grad<'a, T: Dtype>(a: &'a [T], b: &'a [T]) -> (Cow<'a, [T]>, Cow<'a, [T]>) {
    // t = a + b
    (ones_like(a).into(), ones_like(b).into())
}

pub(crate) fn el_sub_grad<'a, T: Dtype>(a: &'a [T], b: &'a [T]) -> (Cow<'a, [T]>, Cow<'a, [T]>) {
    // t = a - b
    (ones_like(a).into(), el_neg(&ones_like(b)).into())
}

pub(crate) fn el_mul_grad<'a, T: Dtype>(a: &'a [T], b: &'a [T]) -> (Cow<'a, [T]>, Cow<'a, [T]>) {
    // t = a * b
    (b.into(), a.into())
}

pub(crate) fn el_div_grad<'a, T: Dtype>(a: &'a [T], b: &'a [T]) -> (Cow<'a, [T]>, Cow<'a, [T]>) {
    // t = a / b
    // dt_da = 1 / b
    // dt_db = -a / (b * b)
    let dt_da = el_inv(b).into();
    let dt_db = el_bin(|(x, y)| -*x / (*y * *y), a, b).into();
    (dt_da, dt_db)
}

pub(crate) fn el_max_grad<'a, T: Dtype>(a: &'a [T], b: &'a [T]) -> (Cow<'a, [T]>, Cow<'a, [T]>) {
    // t = max(a, b)
    // dt_da = 1 if a > b else 0
    // dt_db = 1 if b > a else 0
    let dt_da = el_gt(a, b).into();
    let dt_db = el_gt(b, a).into();
    (dt_da, dt_db)
}

pub(crate) fn el_min_grad<'a, T: Dtype>(a: &'a [T], b: &'a [T]) -> (Cow<'a, [T]>, Cow<'a, [T]>) {
    // t = min(a, b)
    // dt_da = 1 if a < b else 0
    // dt_db = 1 if b < a else 0
    let dt_da = el_lt(a, b).into();
    let dt_db = el_lt(b, a).into();
    (dt_da, dt_db)
}

pub(crate) fn reduce_sum_grad<T: Dtype>(a: &[T]) -> Cow<[T]> {
    // t = sum(a)
    ones_like(a).into()
}

pub(crate) fn el_relu_grad<T: Dtype>(a: &[T]) -> Cow<[T]> {
    // t = relu(a)
    el_pos(a).into()
}
