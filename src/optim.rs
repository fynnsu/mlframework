use crate::dtype::Dtype;
use crate::ops::vec::{el_sub, scalar_mul};
use num::FromPrimitive;
use std::any::type_name;

pub trait Optimizer {
    fn compute<T: Dtype>(&mut self, tid: usize, t_value: &[T], t_grad: &[T]) -> Vec<T>;
}

pub struct GradientDescent {
    pub lr: f32,
}

impl Optimizer for GradientDescent {
    fn compute<T: Dtype>(&mut self, _tid: usize, t_value: &[T], t_grad: &[T]) -> Vec<T> {
        let lr: T = FromPrimitive::from_f32(self.lr)
            .expect(&format!("Failed to cast lr to {}", type_name::<T>()));
        el_sub(t_value, &scalar_mul(lr, t_grad))
    }
}
