// use std::collections::HashMap;

// use crate::{
//     dtype::Dtype,
//     shape::{Const, Shape},
//     tensor::Tensor,
// };

// use super::vec::{
//     el_add, el_bin, el_gt, el_inv, el_lt, el_mul, el_neg, expand_to_shape, ones_like,
// };

// fn update_grad<T: Dtype>(grads: &mut HashMap<usize, Vec<T>>, id: usize, d_dx: Vec<T>) {
//     grads
//         .entry(id)
//         .and_modify(|v| *v = el_add(v, &d_dx))
//         .or_insert(d_dx);
// }

// pub(crate) fn el_add_backward<T: Dtype, S: Shape>(
//     mut grads: &mut HashMap<usize, Vec<T>>,
//     t: &Tensor<T, S>,
//     a: &Tensor<T, S>,
//     b: &Tensor<T, S>,
// ) {
//     // t = a + b
//     let d_dt = grads.get(&t.id).unwrap();
//     let d_da = el_mul(d_dt, &ones_like(a.data.as_ref()));
//     let d_db = el_mul(d_dt, &ones_like(b.data.as_ref()));

//     update_grad(&mut grads, a.id, d_da);
//     update_grad(&mut grads, b.id, d_db);
// }

// pub(crate) fn el_sub_backward<T: Dtype, S: Shape>(
//     mut grads: &mut HashMap<usize, Vec<T>>,
//     t: &Tensor<T, S>,
//     a: &Tensor<T, S>,
//     b: &Tensor<T, S>,
// ) {
//     // t = a - b
//     let d_dt = grads.get(&t.id).unwrap();
//     let d_da = el_mul(d_dt, &ones_like(a.data.as_ref()));
//     let d_db = el_mul(d_dt, &el_neg(&ones_like(b.data.as_ref())));

//     update_grad(&mut grads, a.id, d_da);
//     update_grad(&mut grads, b.id, d_db);
// }

// pub(crate) fn el_mul_backward<T: Dtype, S: Shape>(
//     mut grads: &mut HashMap<usize, Vec<T>>,
//     t: &Tensor<T, S>,
//     a: &Tensor<T, S>,
//     b: &Tensor<T, S>,
// ) {
//     // t = a * b
//     let d_dt = grads.get(&t.id).unwrap();
//     let d_da = el_mul(d_dt, b.data.as_ref());
//     let d_db = el_mul(d_dt, a.data.as_ref());

//     update_grad(&mut grads, a.id, d_da);
//     update_grad(&mut grads, b.id, d_db);
// }

// pub(crate) fn el_div_backward<T: Dtype, S: Shape>(
//     mut grads: &mut HashMap<usize, Vec<T>>,
//     t: &Tensor<T, S>,
//     a: &Tensor<T, S>,
//     b: &Tensor<T, S>,
// ) {
//     // t = a / b
//     let d_dt = grads.get(&t.id).unwrap();

//     let dt_db = el_bin(|(x, y)| -*x / (*y * *y), a.data.as_ref(), b.data.as_ref());
//     let d_da = el_mul(d_dt, &el_inv(b.data.as_ref()));
//     let d_db = el_mul(d_dt, &dt_db);

//     update_grad(&mut grads, a.id, d_da);
//     update_grad(&mut grads, b.id, d_db);
// }

// pub(crate) fn el_max_backward<T: Dtype, S: Shape>(
//     mut grads: &mut HashMap<usize, Vec<T>>,
//     t: &Tensor<T, S>,
//     a: &Tensor<T, S>,
//     b: &Tensor<T, S>,
// ) {
//     // t = max(a, b)
//     let d_dt = grads.get(&t.id).unwrap();

//     let dt_da = el_gt(a.data.as_ref(), b.data.as_ref());
//     let dt_db = el_gt(b.data.as_ref(), a.data.as_ref());
//     let d_da = el_mul(d_dt, &dt_da);
//     let d_db = el_mul(d_dt, &dt_db);

//     update_grad(&mut grads, a.id, d_da);
//     update_grad(&mut grads, b.id, d_db);
// }

// pub(crate) fn el_min_backward<T: Dtype, S: Shape>(
//     mut grads: &mut HashMap<usize, Vec<T>>,
//     t: &Tensor<T, S>,
//     a: &Tensor<T, S>,
//     b: &Tensor<T, S>,
// ) {
//     // t = max(a, b)
//     let d_dt = grads.get(&t.id).unwrap();

//     let dt_da = el_lt(a.data.as_ref(), b.data.as_ref());
//     let dt_db = el_lt(b.data.as_ref(), a.data.as_ref());
//     let d_da = el_mul(d_dt, &dt_da);
//     let d_db = el_mul(d_dt, &dt_db);

//     update_grad(&mut grads, a.id, d_da);
//     update_grad(&mut grads, b.id, d_db);
// }

// pub(crate) fn reduce_sum_backward<T: Dtype, S: Shape>(
//     mut grads: &mut HashMap<usize, Vec<T>>,
//     t: &Tensor<T, (Const<1>,)>,
//     a: &Tensor<T, S>,
// ) {
//     // t = a.sum()
//     let d_dt = grads.get(&t.id).unwrap();
//     let dt_da = ones_like(a.data.as_ref());
//     let d_dt_expanded = expand_to_shape(d_dt, a.data.len());
//     let d_da = el_mul(&d_dt_expanded, &dt_da);

//     update_grad(&mut grads, a.id, d_da);
// }

// impl<T: Dtype> TracedOp<T> for ElReLU<T> {
//     fn backward(&self, t: &Tensor<T>, mut grads: &mut HashMap<usize, Vec<T>>) {
//         // t = max(a, 0)
//         let d_dt = grads.get(&t.id).unwrap();

//         let dt_da = el_pos(self.0.data.as_ref());
//         let d_da = el_mul(d_dt, &dt_da);

//         update_grad(&mut grads, self.0.id, d_da);
//     }

//     fn arg_vec(&self) -> Vec<&Tensor<T>> {
//         vec![&self.0]
//     }
// }
