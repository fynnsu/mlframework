use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::{fmt, usize};
use std::{
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    sync::Mutex,
};

use crate::op::{ElementwiseBinary, Op};
use crate::{dtype::Dtype, op::ElementwiseUnary};

#[derive(Debug)]
pub struct TensorGraph<'a, T: Dtype> {
    pub nodes: HashMap<usize, &'a Tensor<T>>,
    pub edges: HashMap<usize, Vec<usize>>,
}

impl<'a, T: Dtype> TensorGraph<'a, T> {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }
}

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
    pub data: Rc<Vec<T>>,
    _op: Option<Box<Op<T>>>,
    pub id: usize,
}

impl<T: Dtype> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            _op: self._op.clone(),
            id: self.id,
        }
    }
}

impl<T: Dtype + fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({:?}, id={})", self.data.as_ref(), self.id)
    }
}

impl<T: Dtype> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data: Rc::new(data),
            _op: None,
            id: ID_GEN.lock().unwrap().next().unwrap(),
        }
    }

    pub fn new_with_op(data: Vec<T>, op: Op<T>) -> Self {
        Self {
            data: Rc::new(data),
            _op: Some(Box::new(op)),
            id: ID_GEN.lock().unwrap().next().unwrap(),
        }
    }

    pub fn grad(&self) -> HashMap<usize, Vec<T>> {
        let mut grads = HashMap::new();

        assert_eq!(self.data.len(), 1);
        let id_to_tensor_ref = self.build_graph();

        // todo: fix recursion issue
        // Currently diamond pattern graphs could lead to exponential compute time for traversal ordering
        let ordering_map = self.traversal_ordering();
        let mut ordering_vec: Vec<_> = ordering_map.iter().collect();
        ordering_vec.sort_by(|a, b| a.1.cmp(b.1));

        let t_vec: Vec<_> = ordering_vec
            .iter()
            .map(|v| *id_to_tensor_ref.nodes.get(v.0).unwrap())
            .collect();

        grads.insert(self.id, vec![T::one()]);

        for t in t_vec {
            if let Some(op) = &t._op {
                grads = op.as_ref().propogate_grad(grads, t.id);
            }
        }

        grads
    }

    fn _build_graph<'a: 'b, 'b>(&'a self, mut graph: TensorGraph<'b, T>) -> TensorGraph<'b, T> {
        if graph.nodes.contains_key(&self.id) {
            return graph;
        }

        graph.nodes.insert(self.id, &self);
        if let Some(box_op) = &self._op {
            for sub_tensor in box_op.as_ref().iter() {
                graph
                    .edges
                    .entry(sub_tensor.id)
                    .and_modify(|v| {
                        v.push(self.id);
                    })
                    .or_insert(vec![self.id]);
                graph = sub_tensor._build_graph(graph)
            }
        }

        graph
    }

    pub fn build_graph<'a: 'b, 'b>(&'a self) -> TensorGraph<'b, T> {
        let graph = TensorGraph::new();

        self._build_graph(graph)
    }

    // fn _recursive_helper<F, G>(&self, mut f: F, mut args: G) -> (F, G)
    // where
    //     F: FnMut(&Self, G) -> G,
    // {
    //     args = f(self, args);
    //     if let Some(box_op) = &self._op {
    //         for sub_tensor in box_op.as_ref().iter() {
    //             (f, args) = sub_tensor._recursive_helper(f, args);
    //         }
    //     }
    //     (f, args)
    // }

    // pub fn traversal_ordering(&self) -> HashMap<usize, usize> {
    //     let hm = HashMap::new();

    //     let (_, (hm, _)) = self._recursive_helper(
    //         |s: &Self, (mut hm, d): (HashMap<usize, usize>, usize)| {
    //             hm.entry(s._id)
    //                 .and_modify(|v| *v = (*v).max(d))
    //                 .or_insert(d);
    //             (hm, d + 1)
    //         },
    //         (hm, 0),
    //     );
    //     hm
    // }

    fn _traversal_ordering_helper(
        &self,
        mut hm: HashMap<usize, usize>,
        depth: usize,
    ) -> HashMap<usize, usize> {
        hm.entry(self.id)
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
        assert_eq!(self.data.len(), other.data.len());
        Self::new_with_op(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| eb._f(a, b))
                .collect(),
            Op::EB(eb, self, other),
        )
    }

    fn elementwise_unary_op(self, eu: ElementwiseUnary) -> Self {
        Self::new_with_op(
            self.data.iter().map(|a| eu._f(a)).collect(),
            Op::EU(eu, self),
        )
    }

    pub fn relu(self) -> Self {
        self.elementwise_unary_op(ElementwiseUnary::Relu)
    }

    pub fn sum(self) -> Self {
        let s = self.data.iter().fold(T::zero(), |s, x| s + *x);
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
