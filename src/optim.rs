use crate::dtype::Dtype;

pub trait Optimizer {
    fn compute<T: Dtype>(&mut self, tid: usize, t_value: &Vec<T>, t_grad: &Vec<T>) -> Vec<T>;
}
