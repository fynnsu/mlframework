use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;
use std::{
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    sync::Mutex,
};

use crate::op::{ElementwiseBinary, Op};
use crate::{dtype::Dtype, op::ElementwiseUnary};

struct IdGenerator {
    next_id: usize,
}

impl IdGenerator {
    fn new() -> Self {
        IdGenerator { next_id: 0 }
    }
}

impl Iterator for IdGenerator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.next_id;
        self.next_id += 1;
        Some(v)
    }
}

static ID_GEN: Lazy<Mutex<IdGenerator>> = Lazy::new(|| Mutex::new(IdGenerator::new()));

pub struct Tensor<T: Dtype> {
    _data: Rc<Vec<T>>,
    _op: Option<Box<Op<T>>>,
    _id: usize,
}

impl<T: Dtype> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            _data: Rc::clone(&self._data),
            _op: self._op.clone(),
            _id: ID_GEN.lock().unwrap().next().unwrap(),
        }
    }
}

impl<T: Dtype + fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({:?}, id={})", self._data.as_ref(), self._id)
    }
}

impl<T: Dtype> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            _data: Rc::new(data),
            _op: None,
            _id: ID_GEN.lock().unwrap().next().unwrap(),
        }
    }

    pub fn new_with_op(data: Vec<T>, op: Op<T>) -> Self {
        Self {
            _data: Rc::new(data),
            _op: Some(Box::new(op)),
            _id: ID_GEN.lock().unwrap().next().unwrap(),
        }
    }

    fn _propogate_grad(&self, mut grads: HashMap<usize, Vec<T>>) -> HashMap<usize, Vec<T>> {
        grads
    }

    pub fn grad(&self) -> HashMap<usize, Vec<T>> {
        let mut grads = HashMap::new();

        assert_eq!(self._data.len(), 1);
        let id_to_tensor_ref = self.id_to_tensor_ref();

        let ordering_map = self.traversal_ordering();
        let mut ordering_vec: Vec<_> = ordering_map.iter().collect();
        ordering_vec.sort_by(|a, b| a.1.cmp(b.1));

        let t_vec: Vec<_> = ordering_vec
            .iter()
            .map(|v| *id_to_tensor_ref.get(v.0).unwrap())
            .collect();

        grads.insert(self._id, vec![T::one()]);

        for t in t_vec {
            grads = t._propogate_grad(grads);
        }

        grads
    }

    fn _id_to_tensor_ref_helper<'a>(
        &'a self,
        mut hm: HashMap<usize, &'a Self>,
    ) -> HashMap<usize, &Self> {
        hm.insert(self._id, &self);

        if let Some(box_op) = &self._op {
            for sub_tensor in box_op.as_ref().iter() {
                hm = sub_tensor._id_to_tensor_ref_helper(hm)
            }
        }

        hm
    }

    pub fn id_to_tensor_ref(&self) -> HashMap<usize, &Self> {
        let hm = HashMap::new();
        self._id_to_tensor_ref_helper(hm)
    }

    // fn _recursive_helper<F, G, H>(&self, mut f: F, args: G) -> H
    // where
    //     F: FnMut(&Self, G) -> H,
    // {
    //     let mut h = f(&self, args);
    //     if let Some(box_op) = &self._op {
    //         for sub_tensor in box_op.as_ref().iter() {
    //             h = sub_tensor._traversal_ordering_helper(hm, depth + 1)
    //         }
    //     }
    // }

    fn _traversal_ordering_helper(
        &self,
        mut hm: HashMap<usize, usize>,
        depth: usize,
    ) -> HashMap<usize, usize> {
        hm.entry(self._id)
            .and_modify(|v| *v = (*v).max(depth))
            .or_insert(depth);
        if let Some(box_op) = &self._op {
            for sub_tensor in box_op.as_ref().iter() {
                hm = sub_tensor._traversal_ordering_helper(hm, depth + 1)
            }
        }
        hm
    }
    pub fn traversal_ordering(&self) -> HashMap<usize, usize> {
        self._traversal_ordering_helper(HashMap::new(), 0)
    }

    // pub fn traversal_ordering(&self) -> Vec<(Self, usize)> {}
    // fn update_grad_semiring<F>(&self, new_grad: Vec<T>, combine_fn: F)
    // where
    //     F: FnOnce(Vec<T>, Vec<T>),
    // {
    //     let cur_grad = self._data._grad.borrow_mut();
    // }
    // fn update_grad(&self, new_grad: Vec<T>) {
    //     let f = |a, b| inplace_vec_binary(|x, y| *x = *x + *y, a, b);
    //     self.update_grad_semiring(new_grad, f)
    // }
    // fn propogate_grad_semiring<F>(&self, combine_fn: F)
    // where
    //     F: FnOnce

    // pub fn backwards(&mut self) {
    //     assert_eq!(self._data.len(), 1);
    // }

    fn elementwise_binary_op(self, eb: ElementwiseBinary, other: Self) -> Self {
        assert_eq!(self._data.len(), other._data.len());
        Self::new_with_op(
            self._data
                .iter()
                .zip(other._data.iter())
                .map(|(a, b)| eb._f(a, b))
                .collect(),
            Op::EB(eb, self, other),
        )
    }

    fn elementwise_unary_op(self, eu: ElementwiseUnary) -> Self {
        Self::new_with_op(
            self._data.iter().map(|a| eu._f(a)).collect(),
            Op::EU(eu, self),
        )
    }

    pub fn relu(self) -> Self {
        self.elementwise_unary_op(ElementwiseUnary::Relu)
    }

    pub fn sum(self) -> Self {
        let s = self._data.iter().fold(T::zero(), |s, x| s + *x);
        Self::new_with_op(vec![s], Op::Sum(self))
    }
}

impl<T: Dtype> Add for Tensor<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.elementwise_binary_op(ElementwiseBinary::Add, other)
    }
}
impl<T: Dtype> Sub for Tensor<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.elementwise_binary_op(ElementwiseBinary::Sub, other)
    }
}
impl<T: Dtype> Mul for Tensor<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.elementwise_binary_op(ElementwiseBinary::Mul, other)
    }
}
impl<T: Dtype> Div for Tensor<T> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.elementwise_binary_op(ElementwiseBinary::Div, other)
    }
}
