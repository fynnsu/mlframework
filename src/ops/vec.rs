use crate::dtype::Dtype;

pub(crate) fn ones_like<T: Dtype>(a: &[T]) -> Vec<T> {
    vec![T::one(); a.len()]
}
pub(crate) fn zeros_like<T: Dtype>(a: &[T]) -> Vec<T> {
    vec![T::zero(); a.len()]
}
pub(crate) fn ones<T: Dtype>(n: usize) -> Vec<T> {
    vec![T::one(); n]
}

pub(crate) fn dot<T: Dtype>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |s, (a, b)| s + (*a * *b))
}

/// Perform a 2d transpose on an array ref that represents an
/// (m x n) matrix.
pub(crate) fn transpose2d<T: Dtype>(a: &[T], n: usize) -> Vec<T> {
    assert!(a.len() % n == 0);
    (0..n).fold(Vec::with_capacity(a.len()), |mut v, i| {
        v.extend(a.iter().skip(i).step_by(n));
        v
    })
}

#[test]
fn test_transpose2d() {
    let n = 3;
    let a: Vec<_> = (0..15).collect();
    // v = [
    //  0, 1, 2,
    //  3, 4, 5,
    //  6, 7, 8,
    //  9, 10, 11
    //  12, 13, 14
    // ] shape = (m x n)
    let v_transpose = transpose2d(&a, n);
    let target = vec![0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14];
    assert_eq!(v_transpose, target);
}

/// Perform a matmul op between two array refs that represent matrices with the shapes below
/// a: (n, m)
/// b: (m, o)
pub(crate) fn matmul<T: Dtype>(a: &[T], b: &[T], n: usize, m: usize, o: usize) -> Vec<T> {
    assert_eq!(n * m, a.len());
    assert_eq!(m * o, b.len());
    a.chunks(m).fold(Vec::with_capacity(n * o), |v, a_row| {
        let b_t = transpose2d(b, o); // shape = (O, M)
        let v = b_t.chunks(m).fold(v, |mut v, b_col| {
            v.push(dot(a_row, b_col));
            v
        });
        v
    })
}

#[test]
fn test_matmul() {
    let (n, m, o): (usize, usize, usize) = (3, 4, 2);
    let a: Vec<i32> = (0..(n * m) as i32).collect(); // (n, m)
    let b: Vec<i32> = (0..(m * o) as i32).collect(); // (m, o)

    // a = [
    //   0, 1, 2, 3,
    //   4, 5, 6, 7,
    //   8, 9, 10, 11,
    // ]
    // b = [
    //   0, 1,
    //   2, 3,
    //   4, 5,
    //   6, 7
    // ]

    let a_mat_b = matmul(&a, &b, n, m, o);
    let target = vec![28, 34, 76, 98, 124, 162];
    assert_eq!(a_mat_b, target);
}

pub(crate) fn expand_to_shape<T: Dtype>(a: &[T], len: usize) -> Vec<T> {
    assert_eq!(a.len(), 1);
    vec![a[0]; len]
}

pub(crate) fn el_bin<T: Dtype, F>(op: F, a: &[T], b: &[T]) -> Vec<T>
where
    F: Fn((&T, &T)) -> T,
{
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(op).collect()
}

pub(crate) fn el_mul<T: Dtype>(a: &[T], b: &[T]) -> Vec<T> {
    el_bin(|(x, y)| *x * *y, a, b)
}

pub(crate) fn el_add<T: Dtype>(a: &[T], b: &[T]) -> Vec<T> {
    el_bin(|(x, y)| *x + *y, a, b)
}

pub(crate) fn el_sub<T: Dtype>(a: &[T], b: &[T]) -> Vec<T> {
    el_bin(|(x, y)| *x - *y, a, b)
}

pub(crate) fn el_div<T: Dtype>(a: &[T], b: &[T]) -> Vec<T> {
    el_bin(|(x, y)| *x / *y, a, b)
}

pub(crate) fn el_max<T: Dtype>(a: &[T], b: &[T]) -> Vec<T> {
    el_bin(|(x, y)| if *x >= *y { *x } else { *y }, a, b)
}

pub(crate) fn el_min<T: Dtype>(a: &[T], b: &[T]) -> Vec<T> {
    el_bin(|(x, y)| if *x <= *y { *x } else { *y }, a, b)
}

pub(crate) fn el_gt<T: Dtype>(a: &[T], b: &[T]) -> Vec<T> {
    el_bin(|(x, y)| if *x >= *y { T::one() } else { T::zero() }, a, b)
}

pub(crate) fn el_lt<T: Dtype>(a: &[T], b: &[T]) -> Vec<T> {
    el_bin(|(x, y)| if *x <= *y { T::one() } else { T::zero() }, a, b)
}

pub(crate) fn el_unary<T1, T2, F>(op: F, a: &[T1]) -> Vec<T2>
where
    F: Fn(&T1) -> T2,
{
    a.iter().map(op).collect()
}

pub(crate) fn el_neg<T: Dtype>(a: &[T]) -> Vec<T> {
    el_unary(|x| T::neg(*x), a)
}

pub(crate) fn el_relu<T: Dtype>(a: &[T]) -> Vec<T> {
    el_unary(|x| if *x >= T::zero() { *x } else { T::zero() }, a)
}

pub(crate) fn el_inv<T: Dtype>(a: &[T]) -> Vec<T> {
    el_unary(|x| T::one() / *x, a)
}

pub(crate) fn el_pos<T: Dtype>(a: &[T]) -> Vec<T> {
    // Used for relu grad
    // Return 1 if x >= 0, else 0
    el_unary(|x| if *x >= T::zero() { T::one() } else { T::zero() }, a)
}
