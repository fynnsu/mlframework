#[derive(Debug)]
pub struct I<const S: usize>;

pub trait Dim: std::fmt::Debug + 'static {
    fn size() -> usize;
}

impl<const S: usize> Dim for I<S> {
    fn size() -> usize {
        S
    }
}

pub trait Shape: std::fmt::Debug + 'static {
    const NUM_DIMS: usize;
    const NUM_ELS: usize;
    fn strides() -> &'static [usize];
    fn shape() -> &'static [usize];
}

impl Shape for () {
    const NUM_DIMS: usize = 0;
    const NUM_ELS: usize = 0;

    fn strides() -> &'static [usize] {
        &[]
    }
    fn shape() -> &'static [usize] {
        &[]
    }
}

impl<const A: usize> Shape for (I<A>,) {
    const NUM_DIMS: usize = 1;
    const NUM_ELS: usize = A;

    fn strides() -> &'static [usize] {
        &[1]
    }

    fn shape() -> &'static [usize] {
        &[A]
    }
}

impl<const A: usize, const B: usize> Shape for (I<A>, I<B>) {
    const NUM_DIMS: usize = 2;
    const NUM_ELS: usize = { A * B };

    fn strides() -> &'static [usize] {
        &[B, 1]
    }

    fn shape() -> &'static [usize] {
        &[A, B]
    }
}

impl<const A: usize, const B: usize, const C: usize> Shape for (I<A>, I<B>, I<C>) {
    const NUM_DIMS: usize = 3;
    const NUM_ELS: usize = { A * B * C };

    fn strides() -> &'static [usize] {
        &[B * C, C, 1]
    }

    fn shape() -> &'static [usize] {
        &[A, B, C]
    }
}

pub type D1<const N: usize> = (I<N>,);
pub type D2<const N: usize, const M: usize> = (I<N>, I<M>);
pub type D3<const N: usize, const M: usize, const O: usize> = (I<N>, I<M>, I<O>);

pub trait HasNEls<const N: usize> {}

impl<const N: usize> HasNEls<N> for D1<N> {}
impl<const N: usize, const M: usize> HasNEls<{ N * M }> for D2<N, M> {}
impl<const N: usize, const M: usize, const O: usize> HasNEls<{ N * M * O }> for D3<N, M, O> {}

#[macro_export]
macro_rules! s {
    () => ();
    ($d:expr) => {(I<$d>,)};
    ( $($d:expr),+ ) => {($(I<$d>),+)}
}
