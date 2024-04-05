use std::cell::Ref;
use std::collections::{BinaryHeap, HashSet};
use std::fmt::Debug;
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

pub trait TensorTrait: Debug {
    fn process_grad(&self);
    fn parents(&self) -> Vec<TensorBox>;
    fn grad_to_string(&self) -> String;
    fn recompute(&self);
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

    fn grad_to_string(&self) -> String {
        format!("{:?}", self.borrow_grad())
    }

    fn recompute(&self) {
        if let Some(op) = &self.op {
            op.recompute(self)
        }
    }
}

#[derive(Debug)]
pub struct TensorBox<'a> {
    pub(crate) id: usize,
    pub tensor: &'a dyn TensorTrait,
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
        let strides = S::strides();
        let data = self.borrow_value();
        let mut t_str = String::with_capacity(data.len() * 4);
        t_str += &"[".repeat(S::NUM_DIMS);
        t_str += &format!("{:.2?}", data[0]);
        for i in 1..S::NUM_ELS {
            let mut n_match = 0;
            for s in strides[..S::NUM_DIMS - 1].iter() {
                if i % s == 0 {
                    n_match += 1;
                    t_str += "]";
                }
            }
            t_str += ", ";
            t_str += &"[".repeat(n_match);
            t_str += &format!("{:.2?}", data[i]);
        }
        t_str += &"]".repeat(S::NUM_DIMS);
        write!(
            f,
            "Tensor({}, shape={:?}, id={})",
            t_str,
            S::shape(),
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

    pub(crate) fn ancestors(&self) -> HashSet<TensorBox> {
        let mut visited_set = HashSet::new();
        let mut to_visit = vec![TensorBox::new(self.id, self)];

        while let Some(tb) = to_visit.pop() {
            if visited_set.contains(&tb) {
                continue;
            }
            to_visit.extend(tb.tensor.parents());
            visited_set.insert(tb);
        }
        visited_set
    }

    pub fn leaves(&self) -> HashSet<TensorBox> {
        let mut ans = self.ancestors();
        ans.retain(|TensorBox { id: _, tensor: t }| t.parents() == vec![]);
        ans
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

    pub fn recompute(&self) {
        let ancestors = self.ancestors();
        let mut ancestors: Vec<_> = ancestors.iter().collect::<Vec<_>>();
        ancestors.sort();

        for b in ancestors {
            b.tensor.recompute();
        }

        if let Some(op) = &self.op {
            op.recompute(self);
        }
    }

    pub fn replace_data_with(&self, new_data: Vec<T>) {
        {
            assert_eq!(new_data.len(), self.borrow_value().len());
        }
        self.data.replace(new_data);
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

pub fn remove_inputs(tensors: &mut HashSet<TensorBox>, input_ids: &[usize]) {
    tensors.retain(|e| !input_ids.contains(&e.id));
}
