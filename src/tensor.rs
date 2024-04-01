use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::{fmt, usize};

use crate::dtype::Dtype;
// use crate::graph::TensorGraph;
use crate::ops::Op;
use crate::shape::Shape;
use crate::tensor_id::generate_id;
use core::fmt::Debug;

pub struct Tensor<T: Dtype, S: Shape> {
    pub data: Rc<Vec<T>>,
    pub op: Option<Rc<dyn Op<Produces = Tensor<T, S>>>>,
    pub id: usize,
    pub(crate) _shape: PhantomData<S>,
}

#[cfg(test)]
mod tests {
    use crate::shape::Const;

    use super::*;

    #[test]
    fn test_create_tensor_from_vec() {
        let t: Tensor<i32, (Const<4>,)> = Tensor::new(vec![0, 0, 1, 2]);
    }

    #[test]
    fn test_create_tensor_from_tensor() {
        let t = Tensor::new([[2, 3]; 7]);
        let t2 = Tensor::new(t.clone());
        assert_eq!(t._shape, t2._shape);
    }

    #[test]
    fn test_create_tensor_from_1d_array() {
        let t = Tensor::new([2, 9, 8, 7, 8, 2, 3, 0, 0, 0, 1, 2]);
    }

    #[test]
    fn test_create_tensor_from_2d_array() {
        let t = Tensor::new([[2, 9, 8, 7], [8, 2, 3, 0], [0, 0, 1, 2]]);
    }

    #[test]
    fn test_create_tensor_from_3d_array() {
        let t = Tensor::new([[[2, 9], [8, 7]], [[8, 2], [3, 0]], [[0, 0], [1, 2]]]);
        //todo: Test these values better
    }
}

impl<T: Dtype, S: Shape> Clone for Tensor<T, S> {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            op: match &self.op {
                Some(_op) => Some(Rc::clone(&_op)),
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
            data: Rc::new(value),
            op: None,
            id: generate_id(),
            _shape: Default::default(),
        }
    }
    pub(crate) unsafe fn from_rc_vec_and_op_unchecked(
        value: Rc<Vec<T>>,
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

    pub fn new_with_op(
        data: impl Into<Tensor<T, S>>,
        op: Rc<dyn Op<Produces = Tensor<T, S>>>,
    ) -> Self {
        let new_t = data.into();
        Self {
            op: Some(op),
            ..new_t
        }
    }

    // pub fn grad(&self) -> HashMap<usize, Vec<T>> {
    //     let mut grads = HashMap::new();

    //     assert_eq!(self.data.len(), 1);
    //     let id_to_tensor_ref = self.build_graph();

    //     // todo: fix recursion issue
    //     // Currently diamond pattern graphs could lead to exponential compute time for traversal ordering
    //     let ordering_map = self.traversal_ordering();
    //     let mut ordering_vec: Vec<_> = ordering_map.iter().collect();
    //     ordering_vec.sort_by(|a, b| a.1.cmp(b.1));

    //     let t_vec: Vec<_> = ordering_vec
    //         .iter()
    //         .map(|v| *id_to_tensor_ref.nodes.get(v.0).unwrap())
    //         .collect();

    //     grads.insert(self.id, vec![T::one()]);

    //     for t in t_vec {
    //         if let Some(op) = &t.op {
    //             op.as_ref().propogate_grad(&mut grads, t);
    //         }
    //     }

    //     grads
    // }

    // fn _build_graph<'a: 'b, 'b>(&'a self, mut graph: TensorGraph<'b, T>) -> TensorGraph<'b, T> {
    //     if graph.nodes.contains_key(&self.id) {
    //         return graph;
    //     }

    //     graph.nodes.insert(self.id, &self);
    //     if let Some(box_op) = &self.op {
    //         for sub_tensor in box_op.as_ref().iter() {
    //             graph
    //                 .edges
    //                 .entry(sub_tensor.id)
    //                 .and_modify(|v| {
    //                     v.push(self.id);
    //                 })
    //                 .or_insert(vec![self.id]);
    //             graph = sub_tensor._build_graph(graph)
    //         }
    //     }

    //     graph
    // }

    // pub fn build_graph<'a: 'b, 'b>(&'a self) -> TensorGraph<'b, T> {
    //     let graph = TensorGraph::new();

    //     self._build_graph(graph)
    // }

    // fn _traversal_ordering_helper(
    //     &self,
    //     mut hm: HashMap<usize, usize>,
    //     depth: usize,
    // ) -> HashMap<usize, usize> {
    //     hm.entry(self.id)
    //         .and_modify(|v| *v = (*v).max(depth))
    //         .or_insert(depth);
    //     if let Some(box_op) = &self.op {
    //         for sub_tensor in box_op.as_ref().iter() {
    //             hm = sub_tensor._traversal_ordering_helper(hm, depth + 1)
    //         }
    //     }
    //     hm
    // }
    // pub fn traversal_ordering(&self) -> HashMap<usize, usize> {
    //     self._traversal_ordering_helper(HashMap::new(), 0)
    // }
}
