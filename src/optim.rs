use crate::dtype::Dtype;

pub trait Optimizer {
    fn compute<T: Dtype>(&mut self, tid: usize, t_value: &[T], t_grad: &[T]) -> Vec<T>;
}
