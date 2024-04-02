use super::vec::{el_bin, el_gt, el_inv, el_lt, el_neg, el_pos, ones_like};
use crate::dtype::Dtype;
use std::borrow::Cow;

pub(crate) fn el_add_grad<'a, T: Dtype>(
    a: &'a Vec<T>,
    b: &'a Vec<T>,
) -> (Cow<'a, Vec<T>>, Cow<'a, Vec<T>>) {
    // t = a + b
    (Cow::Owned(ones_like(a)), Cow::Owned(ones_like(b)))
}

pub(crate) fn el_sub_grad<'a, T: Dtype>(
    a: &'a Vec<T>,
    b: &'a Vec<T>,
) -> (Cow<'a, Vec<T>>, Cow<'a, Vec<T>>) {
    // t = a - b
    (Cow::Owned(ones_like(a)), Cow::Owned(el_neg(&ones_like(b))))
}

pub(crate) fn el_mul_grad<'a, T: Dtype>(
    a: &'a Vec<T>,
    b: &'a Vec<T>,
) -> (Cow<'a, Vec<T>>, Cow<'a, Vec<T>>) {
    // t = a * b
    (Cow::Borrowed(b), Cow::Borrowed(a))
}

pub(crate) fn el_div_grad<'a, T: Dtype>(
    a: &'a Vec<T>,
    b: &'a Vec<T>,
) -> (Cow<'a, Vec<T>>, Cow<'a, Vec<T>>) {
    // t = a / b
    // dt_da = 1 / b
    // dt_db = -a / (b * b)
    let dt_da = Cow::Owned(el_inv(b));
    let dt_db = Cow::Owned(el_bin(|(x, y)| -*x / (*y * *y), a, b));
    (dt_da, dt_db)
}

pub(crate) fn el_max_grad<'a, T: Dtype>(
    a: &'a Vec<T>,
    b: &'a Vec<T>,
) -> (Cow<'a, Vec<T>>, Cow<'a, Vec<T>>) {
    // t = max(a, b)
    // dt_da = 1 if a > b else 0
    // dt_db = 1 if b > a else 0
    let dt_da = Cow::Owned(el_gt(a, b));
    let dt_db = Cow::Owned(el_gt(b, a));
    (dt_da, dt_db)
}

pub(crate) fn el_min_grad<'a, T: Dtype>(
    a: &'a Vec<T>,
    b: &'a Vec<T>,
) -> (Cow<'a, Vec<T>>, Cow<'a, Vec<T>>) {
    // t = min(a, b)
    // dt_da = 1 if a < b else 0
    // dt_db = 1 if b < a else 0
    let dt_da = Cow::Owned(el_lt(a, b));
    let dt_db = Cow::Owned(el_lt(b, a));
    (dt_da, dt_db)
}

pub(crate) fn reduce_sum_grad<'a, T: Dtype>(a: &'a Vec<T>) -> (Cow<'a, Vec<T>>) {
    // t = sum(a)
    Cow::Owned(ones_like(a))
}

pub(crate) fn el_relu_grad<'a, T: Dtype>(a: &'a Vec<T>) -> (Cow<'a, Vec<T>>) {
    // t = relu(a)
    Cow::Owned(el_pos(a))
}
