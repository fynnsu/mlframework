use std::cell::{Ref, RefCell};
use std::ops::Deref;

use crate::dtype::Dtype;
use crate::ops::vec::el_add;

#[derive(Debug)]
pub(crate) struct TensorData<T: Dtype> {
    value: Vec<T>,
    grad: RefCell<Option<Vec<T>>>,
}

impl<T: Dtype> TensorData<T> {
    pub(crate) fn new(value: Vec<T>) -> Self {
        Self {
            value,
            grad: RefCell::new(None),
        }
    }

    pub(crate) fn grad_ref(&self) -> Ref<Option<Vec<T>>> {
        self.grad.borrow()
    }

    pub(crate) fn update_grad(&self, new_grad: Vec<T>) {
        let t = match self.grad.borrow().as_ref() {
            Some(g) => el_add(g, &new_grad),
            None => new_grad,
        };
        *self.grad.borrow_mut() = Some(t);
    }
}

impl<T: Dtype> Deref for TensorData<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: Dtype> From<Vec<T>> for TensorData<T> {
    fn from(value: Vec<T>) -> Self {
        Self {
            value,
            grad: RefCell::new(None),
        }
    }
}
