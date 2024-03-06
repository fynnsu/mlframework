use std::collections::HashMap;
use std::rc::Rc;
use std::{fmt, usize};

use crate::dtype::Dtype;
use crate::graph::TensorGraph;
use crate::ops::Op;
use crate::tensor_id::generate_id;

pub struct Tensor<T: Dtype> {
    pub data: Rc<Vec<T>>,
    pub op: Option<Rc<Op<T>>>,
    pub id: usize,
    pub shape: Vec<usize>,
}

pub trait ToVecAndShape<T> {
    fn to_vec_and_shape(self) -> (Vec<T>, Vec<usize>);
}

impl<T: Dtype> ToVecAndShape<T> for Vec<T> {
    fn to_vec_and_shape(self) -> (Vec<T>, Vec<usize>) {
        let l = self.len();
        (self, vec![l])
    }
}

impl<T: Dtype, const N: usize> ToVecAndShape<T> for [T; N] {
    fn to_vec_and_shape(self) -> (Vec<T>, Vec<usize>) {
        (Vec::from(self), vec![N])
    }
}

impl<T: Dtype, const N: usize, const M: usize> ToVecAndShape<T> for [[T; M]; N] {
    fn to_vec_and_shape(self) -> (Vec<T>, Vec<usize>) {
        (Vec::from(self.concat()), vec![N, M])
    }
}

impl<T: Dtype, const N: usize, const M: usize, const Q: usize> ToVecAndShape<T>
    for [[[T; Q]; M]; N]
{
    fn to_vec_and_shape(self) -> (Vec<T>, Vec<usize>) {
        (Vec::from(self.concat().concat()), vec![N, M, Q])
    }
}

impl<T: Dtype, const N: usize, const M: usize, const Q: usize, const R: usize> ToVecAndShape<T>
    for [[[[T; R]; Q]; M]; N]
{
    fn to_vec_and_shape(self) -> (Vec<T>, Vec<usize>) {
        (Vec::from(self.concat().concat().concat()), vec![N, M, Q, R])
    }
}

impl<T: Dtype> ToVecAndShape<T> for Tensor<T> {
    fn to_vec_and_shape(self) -> (Vec<T>, Vec<usize>) {
        (self.data.as_ref().to_owned(), self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tensor_from_vec() {
        let t = Tensor::new(vec![0, 0, 1, 2]);
        assert_eq!(t.shape, vec![4])
    }

    #[test]
    fn test_create_tensor_from_tensor() {
        let t = Tensor::new([[2, 3]; 7]);
        let t2 = Tensor::new(t.clone());
        assert_eq!(t.shape, t2.shape);
    }

    #[test]
    fn test_create_tensor_from_1d_array() {
        let t = Tensor::new([2, 9, 8, 7, 8, 2, 3, 0, 0, 0, 1, 2]);
        assert_eq!(t.shape, vec![12])
    }

    #[test]
    fn test_create_tensor_from_2d_array() {
        let t = Tensor::new([[2, 9, 8, 7], [8, 2, 3, 0], [0, 0, 1, 2]]);
        assert_eq!(t.shape, vec![3, 4])
    }

    #[test]
    fn test_create_tensor_from_3d_array() {
        let t = Tensor::new([[[2, 9], [8, 7]], [[8, 2], [3, 0]], [[0, 0], [1, 2]]]);
        assert_eq!(t.shape, vec![3, 2, 2])
    }

    #[test]
    fn test_create_tensor_from_4d_array() {
        let t = Tensor::new([[[[2, 9], [8, 7]], [[8, 2], [3, 0]], [[0, 0], [1, 2]]]]);
        assert_eq!(t.shape, vec![1, 3, 2, 2])
    }
}

impl<T: Dtype> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            op: match &self.op {
                Some(_op) => Some(Rc::clone(&_op)),
                None => None,
            },
            id: self.id,
            shape: self.shape.clone(),
        }
    }
}

impl<T: Dtype + fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({:?}, shape={:?}, id={})",
            self.data.as_ref(),
            self.shape,
            self.id
        )
    }
}

impl<T: Dtype> Tensor<T> {
    pub fn new(data: impl ToVecAndShape<T>) -> Self {
        let (v, s) = data.to_vec_and_shape();
        Self {
            data: Rc::new(v),
            // _op: None,
            op: None,
            id: generate_id(),
            shape: s,
        }
    }

    pub fn new_with_op(data: impl ToVecAndShape<T>, op: Op<T>) -> Self {
        let (v, s) = data.to_vec_and_shape();
        Self {
            data: Rc::new(v),
            op: Some(Rc::new(op)),
            id: generate_id(),
            shape: s,
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
            if let Some(op) = &t.op {
                op.as_ref().propogate_grad(&mut grads, t);
            }
        }

        grads
    }

    fn _build_graph<'a: 'b, 'b>(&'a self, mut graph: TensorGraph<'b, T>) -> TensorGraph<'b, T> {
        if graph.nodes.contains_key(&self.id) {
            return graph;
        }

        graph.nodes.insert(self.id, &self);
        if let Some(box_op) = &self.op {
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

    fn _traversal_ordering_helper(
        &self,
        mut hm: HashMap<usize, usize>,
        depth: usize,
    ) -> HashMap<usize, usize> {
        hm.entry(self.id)
            .and_modify(|v| *v = (*v).max(depth))
            .or_insert(depth);
        if let Some(box_op) = &self.op {
            for sub_tensor in box_op.as_ref().iter() {
                hm = sub_tensor._traversal_ordering_helper(hm, depth + 1)
            }
        }
        hm
    }
    pub fn traversal_ordering(&self) -> HashMap<usize, usize> {
        self._traversal_ordering_helper(HashMap::new(), 0)
    }
}
