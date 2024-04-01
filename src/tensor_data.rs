use std::ops::Deref;

use crate::dtype::Dtype;

#[derive(Debug)]
pub(crate) struct TensorData<T: Dtype> {
    value: Vec<T>,
    grad: Option<Vec<T>>,
}

impl<T: Dtype> TensorData<T> {
    pub(crate) fn new(value: Vec<T>) -> Self {
        Self { value, grad: None }
    }
}

// impl<T: Dtype> From<&TensorData<T>> for &Vec<T> {
//     fn from(value: &TensorData<T>) -> Self {
//         &value.value
//     }
// }

impl<T: Dtype> Deref for TensorData<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: Dtype> From<Vec<T>> for TensorData<T> {
    fn from(value: Vec<T>) -> Self {
        Self { value, grad: None }
    }
}
