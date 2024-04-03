use std::cell::Ref;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::rc::Rc;
use std::{fmt, usize};

use crate::dtype::Dtype;
// use crate::graph::TensorGraph;
use crate::ops::Op;
use crate::optim::Optimizer;
use crate::shape::{Const, Shape};
use crate::tensor_data::TensorData;
use crate::tensor_id::generate_id;

pub struct Tensor<T: Dtype, S: Shape> {
    pub(crate) data: Rc<TensorData<T>>,
    pub(crate) op: Option<Rc<dyn Op<Produces = Tensor<T, S>>>>,
    pub id: usize,
    pub(crate) _shape: PhantomData<S>,
}

pub(crate) trait TensorTrait {
    fn process_grad(&self);
    fn parents(&self) -> Vec<TensorBox>;
}
impl<T: Dtype, S: Shape> TensorTrait for Tensor<T, S> {
    fn process_grad(&self) {
        if let Some(op) = self.op.as_ref() {
            op.propogate_grad(self);
        }
    }

    fn parents(&self) -> Vec<TensorBox> {
        match &self.op {
            Some(o) => o.operands(),
            None => vec![],
        }
    }
}

pub(crate) struct TensorBox<'a> {
    pub(crate) id: usize,
    pub(crate) tensor: &'a dyn TensorTrait,
}

impl<'a> TensorBox<'a> {
    pub(crate) fn new(id: usize, tensor: &'a dyn TensorTrait) -> Self {
        Self { id, tensor }
    }
}

impl<'a> PartialEq for TensorBox<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<'a> Eq for TensorBox<'a> {}

impl<'a> PartialOrd for TensorBox<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for TensorBox<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
impl<'a> Hash for TensorBox<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use crate::shape::Const;

    use super::*;

    #[test]
    fn test_create_tensor_from_vec() {
        let _t: Tensor<i32, (Const<4>,)> = Tensor::new(vec![0, 0, 1, 2]);
    }

    #[test]
    fn test_create_tensor_from_tensor() {
        let t = Tensor::new([[2, 3]; 7]);
        let t2 = Tensor::new(t.clone());
        assert_eq!(t._shape, t2._shape);
    }

    #[test]
    fn test_create_tensor_from_1d_array() {
        let _t = Tensor::new([2, 9, 8, 7, 8, 2, 3, 0, 0, 0, 1, 2]);
    }

    #[test]
    fn test_create_tensor_from_2d_array() {
        let _t = Tensor::new([[2, 9, 8, 7], [8, 2, 3, 0], [0, 0, 1, 2]]);
    }

    #[test]
    fn test_create_tensor_from_3d_array() {
        let _t = Tensor::new([[[2, 9], [8, 7]], [[8, 2], [3, 0]], [[0, 0], [1, 2]]]);
        //todo: Test these values better
    }
}

impl<T: Dtype, S: Shape> Clone for Tensor<T, S> {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            op: match &self.op {
                Some(_op) => Some(Rc::clone(_op)),
                None => None,
            },
            id: self.id,
            _shape: Default::default(),
        }
    }
}

impl<T: Dtype + fmt::Debug, S: Shape + fmt::Debug> fmt::Debug for Tensor<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({:?}, shape={:?}, id={})",
            self.data.as_ref(),
            std::any::type_name::<S>(),
            self.id
        )
    }
}

impl<T: Dtype, S: Shape> Tensor<T, S> {
    pub(crate) unsafe fn from_vec_unchecked(value: Vec<T>) -> Self {
        Self {
            data: Rc::new(TensorData::new(value)),
            op: None,
            id: generate_id(),
            _shape: Default::default(),
        }
    }
    pub(crate) unsafe fn from_rc_td_and_op_unchecked(
        value: Rc<TensorData<T>>,
        op: Rc<dyn Op<Produces = Tensor<T, S>>>,
    ) -> Self {
        Self {
            data: value,
            op: Some(op),
            id: generate_id(),
            _shape: Default::default(),
        }
    }
    pub fn new(data: impl Into<Tensor<T, S>>) -> Self {
        data.into()
    }

    pub(crate) fn borrow_value(&self) -> Ref<Vec<T>> {
        self.data.value_ref()
    }

    pub(crate) fn borrow_grad(&self) -> Ref<Option<Vec<T>>> {
        self.data.grad_ref()
    }

    pub(crate) fn update_grad(&self, new_grad: Vec<T>) {
        self.data.update_grad(new_grad);
    }

    fn apply_grad<Opt: Optimizer>(&mut self, optim: &mut Opt) {
        let new_value = {
            let t_grad = self.borrow_grad();
            if let Some(t_grad) = t_grad.as_ref() {
                let t_value = self.borrow_value();
                Some(optim.compute(self.id, &t_value, t_grad))
            } else {
                None
            }
        };
        if let Some(new_value) = new_value {
            // Broken up like this to ensure the borrows above are out of scope before replace is called
            self.data.replace(new_value);
        }
    }

    pub(crate) fn new_with_op(
        data: impl Into<Tensor<T, S>>,
        op: Rc<dyn Op<Produces = Tensor<T, S>>>,
    ) -> Self {
        let new_t = data.into();
        Self {
            op: Some(op),
            ..new_t
        }
    }
}

impl<T: Dtype> Tensor<T, (Const<1>,)> {
    pub fn backward(&self) {
        self.update_grad(vec![T::one()]);
        let mut heap = BinaryHeap::new();
        let mut set = HashSet::new();
        let b = TensorBox::new(self.id, self);
        set.insert(b.id);
        heap.push(b);
        while let Some(TensorBox { id: _, tensor: t }) = heap.pop() {
            t.process_grad();
            for parent in t.parents() {
                if !set.contains(&parent.id) {
                    set.insert(parent.id);
                    heap.push(parent);
                }
            }
        }
    }
}
