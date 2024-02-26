use std::{ops::Add, rc::Rc};

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
    fn add(self, other: Self) -> Tensor<T> {
        assert!(self._data.len() == other._data.len());
        Tensor {
            _data: Rc::new(
                self._data
                    .iter()
                    .zip(other._data.iter())
                    .map(|(a, b)| *a + *b)
                    .collect(),
            ),
        }
    }
}
