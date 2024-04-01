use num::Signed;

pub trait Dtype: Copy + Signed + PartialOrd<Self> + std::fmt::Debug + 'static {}
// where Self: std::marker::Sized

impl<T> Dtype for T where T: Copy + Signed + PartialOrd<T> + std::fmt::Debug + 'static {}
