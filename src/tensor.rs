use std::{
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

#[derive(Debug)]
pub struct Tensor<T> {
    pub _data: Rc<Vec<T>>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            _data: Rc::new(data),
        }
    }

    fn elementwise_op<F>(self, f: F, other: Self) -> Self
    where
        F: Fn((&T, &T)) -> T,
    {
        assert!(self._data.len() == other._data.len());
        Self {
            _data: Rc::new(self._data.iter().zip(other._data.iter()).map(f).collect()),
        }
    }
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            _data: Rc::clone(&self._data),
        }
    }
}

impl<T: Add<Output = T> + Copy> Add for Tensor<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.elementwise_op(|(a, b)| *a + *b, other)
    }
}
impl<T: Sub<Output = T> + Copy> Sub for Tensor<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.elementwise_op(|(a, b)| *a - *b, other)
    }
}
impl<T: Mul<Output = T> + Copy> Mul for Tensor<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.elementwise_op(|(a, b)| *a * *b, other)
    }
}
impl<T: Div<Output = T> + Copy> Div for Tensor<T> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.elementwise_op(|(a, b)| *a / *b, other)
    }
}
