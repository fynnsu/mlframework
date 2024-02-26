use std::ops::{Add, Div, Mul, Sub};

pub trait Dtype:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
where
    Self: std::marker::Sized,
{
}

impl<T> Dtype for T where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{
}
