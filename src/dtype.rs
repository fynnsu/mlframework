use num::Num;

pub trait Dtype: Copy + Num + PartialOrd<Self> {}
// where Self: std::marker::Sized

impl<T> Dtype for T where T: Copy + Num + PartialOrd<T> {}
