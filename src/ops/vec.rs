use crate::dtype::Dtype;

pub(crate) fn ones_like<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    vec![T::one(); a.len()]
}
pub(crate) fn zeros_like<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    vec![T::zero(); a.len()]
}
pub(crate) fn ones<T: Dtype>(n: usize) -> Vec<T> {
    vec![T::one(); n]
}

pub(crate) fn expand_to_shape<T: Dtype>(a: &Vec<T>, len: usize) -> Vec<T> {
    assert_eq!(a.len(), 1);
    vec![a[0]; len]
}

pub(crate) fn el_bin<T: Dtype, F>(op: F, a: &Vec<T>, b: &Vec<T>) -> Vec<T>
where
    F: Fn((&T, &T)) -> T,
{
    a.iter().zip(b.iter()).map(op).collect()
}

pub(crate) fn el_mul<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| *x * *y, a, b)
}

pub(crate) fn el_add<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| *x + *y, a, b)
}

pub(crate) fn el_sub<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| *x - *y, a, b)
}

pub(crate) fn el_div<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| *x / *y, a, b)
}

pub(crate) fn el_max<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| if *x >= *y { *x } else { *y }, a, b)
}

pub(crate) fn el_min<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| if *x <= *y { *x } else { *y }, a, b)
}

pub(crate) fn el_gt<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| if *x >= *y { T::one() } else { T::zero() }, a, b)
}

pub(crate) fn el_lt<T: Dtype>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    el_bin(|(x, y)| if *x <= *y { T::one() } else { T::zero() }, a, b)
}

pub(crate) fn el_unary<T: Dtype, F>(op: F, a: &Vec<T>) -> Vec<T>
where
    F: Fn(&T) -> T,
{
    a.iter().map(op).collect()
}

pub(crate) fn el_neg<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    el_unary(|x| T::neg(*x), a)
}

pub(crate) fn el_relu<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    el_unary(|x| if *x >= T::zero() { *x } else { T::zero() }, a)
}

pub(crate) fn el_inv<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    el_unary(|x| T::one() / *x, a)
}

pub(crate) fn el_pos<T: Dtype>(a: &Vec<T>) -> Vec<T> {
    // Used for relu grad
    // Return 1 if x >= 0, else 0
    el_unary(|x| if *x >= T::zero() { T::one() } else { T::zero() }, a)
}
