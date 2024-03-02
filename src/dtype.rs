use num::Signed;

pub trait Dtype: Copy + Signed + PartialOrd<Self> {}
// where Self: std::marker::Sized

impl<T> Dtype for T where T: Copy + Signed + PartialOrd<T> {}
