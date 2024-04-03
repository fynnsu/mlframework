use std::borrow::BorrowMut;
use std::cell::{Ref, RefCell};
use std::ops::Deref;

use crate::dtype::Dtype;
use crate::ops::vec::el_add;

#[derive(Debug)]
pub(crate) struct TensorData<T: Dtype> {
    inner: RefCell<TensorDataInner<T>>,
}

#[derive(Debug)]
pub(crate) struct TensorDataInner<T: Dtype> {
    value: Vec<T>,
    grad: Option<Vec<T>>,
}

impl<T: Dtype> TensorData<T> {
    pub(crate) fn new(value: Vec<T>) -> Self {
        Self {
            inner: RefCell::new(TensorDataInner { value, grad: None }),
        }
    }

    pub(crate) fn replace(&self, value: Vec<T>) {
        let mut inner = self.inner.borrow_mut();
        inner.value = value;
        inner.grad = None;
    }

    pub(crate) fn grad_ref(&self) -> Ref<Option<Vec<T>>> {
        Ref::map(self.inner.borrow(), |t| &t.grad)
    }

    pub(crate) fn value_ref(&self) -> Ref<Vec<T>> {
        Ref::map(self.inner.borrow(), |t| &t.value)
    }

    pub(crate) fn update_grad(&self, new_grad: Vec<T>) {
        let t = {
            match self.grad_ref().deref() {
                Some(g) => el_add(g, &new_grad),
                None => new_grad,
            }
        };
        self.inner.borrow_mut().grad = Some(t);
    }
}

impl<T: Dtype> From<Vec<T>> for TensorData<T> {
    fn from(value: Vec<T>) -> Self {
        Self::new(value)
    }
}
