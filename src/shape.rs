#[derive(Debug)]
pub struct Const<const S: usize>;

pub trait Dim: std::fmt::Debug + 'static {
    fn size(&self) -> usize;
}

impl<const S: usize> Dim for Const<S> {
    fn size(&self) -> usize {
        S
    }
}

impl Dim for usize {
    fn size(&self) -> usize {
        *self
    }
}

pub trait Shape: std::fmt::Debug + 'static {
    const NUM_DIMS: usize;
    fn strides(&self) -> [usize; Self::NUM_DIMS];
}

impl Shape for () {
    const NUM_DIMS: usize = 0;

    fn strides(&self) -> [usize; Self::NUM_DIMS] {
        [0; 0]
    }
}

impl<const A: usize> Shape for (Const<A>,) {
    const NUM_DIMS: usize = 1;

    fn strides(&self) -> [usize; Self::NUM_DIMS] {
        [1]
    }
}

impl<const A: usize, const B: usize> Shape for (Const<A>, Const<B>) {
    const NUM_DIMS: usize = 2;

    fn strides(&self) -> [usize; Self::NUM_DIMS] {
        [B, 1]
    }
}

impl<const A: usize, const B: usize, const C: usize> Shape for (Const<A>, Const<B>, Const<C>) {
    const NUM_DIMS: usize = 3;

    fn strides(&self) -> [usize; Self::NUM_DIMS] {
        [self.1.size() * self.2.size(), self.2.size(), 1]
    }
}

pub type D1<const N: usize> = (Const<N>,);
pub type D2<const N: usize, const M: usize> = (Const<N>, Const<M>);
pub type D3<const N: usize, const M: usize, const O: usize> = (Const<N>, Const<M>, Const<O>);
