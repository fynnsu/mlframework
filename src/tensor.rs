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
    pub fn new(data: Vec<T>) -> Self {
        let l = data.len();
        Self {
            data: Rc::new(data),
            // _op: None,
            op: None,
            id: generate_id(),
            shape: vec![l],
        }
    }

    pub fn new_with_op(data: Vec<T>, op: Op<T>) -> Self {
        let l = data.len();
        Self {
            data: Rc::new(data),
            op: Some(Rc::new(op)),
            id: generate_id(),
            shape: vec![l],
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
