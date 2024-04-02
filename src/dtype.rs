use num::Signed;

pub trait Dtype: Copy + Signed + PartialOrd<Self> + std::fmt::Debug + 'static {}

impl<T> Dtype for T where T: Copy + Signed + PartialOrd<T> + std::fmt::Debug + 'static {}
