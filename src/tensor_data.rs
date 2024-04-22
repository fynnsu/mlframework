use std::cell::{Ref, RefCell};
use std::rc::Rc;

use crate::dtype::Dtype;
use crate::ops::vec::el_add;

#[derive(Debug, Clone)]
pub(crate) struct TensorData<T: Dtype> {
    inner: Rc<RefCell<TensorDataInner<T>>>,
}

#[derive(Debug)]
pub(crate) enum TensorDataInner<T: Dtype> {
    ValueWithGradOption { value: Vec<T>, grad: Option<Vec<T>> },
    Value { value: Vec<T> },
}

use TensorDataInner::*;

impl<T: Dtype> TensorData<T> {
    pub(crate) fn new(value: Vec<T>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(ValueWithGradOption { value, grad: None })),
        }
    }

    pub(crate) fn new_without_grad(value: Vec<T>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Value { value })),
        }
    }

    pub(crate) fn replace(&self, new_value: Vec<T>) {
        match *self.inner.borrow_mut() {
            ValueWithGradOption {
                ref mut value,
                ref mut grad,
            } => {
                *value = new_value;
                *grad = None;
            }
            Value { ref mut value } => *value = new_value,
        }
    }

    pub(crate) fn grad_ref(&self) -> Ref<Option<Vec<T>>> {
        Ref::map(self.inner.borrow(), |t| match t {
            ValueWithGradOption { value: _, ref grad } => grad,
            Value { value: _ } => &None,
        })
    }

    pub(crate) fn value_ref(&self) -> Ref<Vec<T>> {
        Ref::map(self.inner.borrow(), |t| match t {
            ValueWithGradOption { ref value, grad: _ } | Value { ref value } => value,
        })
    }

    pub(crate) fn update_grad(&self, new_grad: Vec<T>) {
        match *self.inner.borrow_mut() {
            Value { value: _ } => {
                panic!("Update grad called on TensorData::Value")
            }
            ValueWithGradOption {
                value: _,
                grad: ref mut g,
            } => {
                let new_g = match g {
                    Some(cur_g) => el_add(cur_g, &new_grad),
                    None => new_grad,
                };
                *g = Some(new_g)
            }
        };
    }
}

impl<T: Dtype> From<Vec<T>> for TensorData<T> {
    fn from(value: Vec<T>) -> Self {
        Self::new(value)
    }
}
