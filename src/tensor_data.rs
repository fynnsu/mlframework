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
    pub(crate) fn new(value: Vec<T>, requires_grad: bool) -> Self {
        Self {
            inner: Rc::new(RefCell::new(if requires_grad {
                ValueWithGradOption { value, grad: None }
            } else {
                Value { value }
            })),
        }
    }

    pub(crate) unsafe fn add_grad_field(&self) {
        self.inner.replace_with(|tdi| {
            let v = match tdi {
                Value { value } => value,
                ValueWithGradOption { value: _, grad: _ } => {
                    panic!("TensorData already has grad field.")
                }
            };
            ValueWithGradOption {
                value: v.clone(),
                grad: None,
            }
        });
    }

    pub(crate) fn has_grad_field(&self) -> bool {
        match *self.inner.borrow() {
            Value { value: _ } => false,
            ValueWithGradOption { value: _, grad: _ } => true,
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

impl<T: Dtype> TensorDataInner<T> {
    fn replace_with_grad_variant(self) -> Self {
        match self {
            Value { value } => ValueWithGradOption { value, grad: None },
            ValueWithGradOption { value: _, grad: _ } => {
                panic!("TensorData already has grad field.")
            }
        }
    }
}
