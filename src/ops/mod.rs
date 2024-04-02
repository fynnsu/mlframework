mod grad;
mod tensor;
pub(crate) mod vec;

pub trait Op: std::fmt::Debug {
    type Produces;
    fn propogate_grad(&self, t: &Self::Produces);
    fn forward(self) -> Self::Produces;
}

// use std::collections::HashMap;

// use crate::{
//     dtype::Dtype,
//     ops::{
//         grad::{
//             el_add_backward, el_div_backward, el_max_backward, el_min_backward, el_mul_backward,
//             el_sub_backward, reduce_sum_backward,
//         },
//         vec::{el_add, el_div, el_max, el_min, el_mul, el_sub},
//     },
//     tensor::{Tensor, TensorTrait},
// };

// #[derive(Debug, Clone)]
// pub enum Op<T: Dtype> {
//     ElAdd(Box<dyn TensorTrait<T>>, Box<dyn TensorTrait<T>>),
//     ElSub(Box<dyn TensorTrait<T>>, Box<dyn TensorTrait<T>>),
//     ElMul(Box<dyn TensorTrait<T>>, Box<dyn TensorTrait<T>>),
//     ElDiv(Box<dyn TensorTrait<T>>, Box<dyn TensorTrait<T>>),
//     ElMax(Box<dyn TensorTrait<T>>, Box<dyn TensorTrait<T>>),
//     ElMin(Box<dyn TensorTrait<T>>, Box<dyn TensorTrait<T>>),
//     ReduceSum(Box<dyn TensorTrait<T>>),
// }

// impl<T: Dtype> Op<T> {
// pub fn forward(self) -> Tensor<T> {
//     use Op::*;
//     match &self {
//         ElAdd(a, b) => Tensor::new_with_op(el_add(a.data.as_ref(), b.data.as_ref()), self),
//         ElSub(a, b) => Tensor::new_with_op(el_sub(a.data.as_ref(), b.data.as_ref()), self),
//         ElMul(a, b) => Tensor::new_with_op(el_mul(a.data.as_ref(), b.data.as_ref()), self),
//         ElDiv(a, b) => Tensor::new_with_op(el_div(a.data.as_ref(), b.data.as_ref()), self),
//         ElMax(a, b) => Tensor::new_with_op(el_max(a.data.as_ref(), b.data.as_ref()), self),
//         ElMin(a, b) => Tensor::new_with_op(el_min(a.data.as_ref(), b.data.as_ref()), self),
//         ReduceSum(a) => {
//             Tensor::new_with_op(vec![a.data.iter().fold(T::zero(), |s, x| s + *x)], self)
//         }
//     }
// }

//     pub fn propogate_grad(&self, mut grads: &mut HashMap<usize, Vec<T>>, t: &dyn TensorTrait<T>) {
//         use Op::*;
//         match self {
//             ElAdd(a, b) => el_add_backward(&mut grads, t, a, b),
//             ElSub(a, b) => el_sub_backward(&mut grads, t, a, b),
//             ElMul(a, b) => el_mul_backward(&mut grads, t, a, b),
//             ElDiv(a, b) => el_div_backward(&mut grads, t, a, b),
//             ElMax(a, b) => el_max_backward(&mut grads, t, a, b),
//             ElMin(a, b) => el_min_backward(&mut grads, t, a, b),
//             ReduceSum(a) => reduce_sum_backward(&mut grads, t, a),
//             // ElMax(a, b) => el_max_backward(&mut grads, t, a, b),
//             // ElMin(a, b) => el_min_backward(&mut grads, t, a, b),
//         }
//     }
// }
//     pub fn iter(&self) -> impl Iterator<Item = &Tensor<T>> {
//         use Op::*;
//         match self {
//             ElAdd(a, b) | ElSub(a, b) | ElMul(a, b) | ElDiv(a, b) | ElMax(a, b) | ElMin(a, b) => {
//                 vec![a, b].into_iter()
//             }
//             ReduceSum(a) => vec![a].into_iter(),
//         }
//     }
// }
