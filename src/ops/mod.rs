mod grad;
mod tensor;
pub(crate) mod vec;

use crate::tensor::TensorBox;

pub(crate) trait Op: std::fmt::Debug {
    type Produces;
    fn propogate_grad(&self, t: &Self::Produces);
    fn forward(self) -> Self::Produces;
    fn operands(&self) -> Vec<TensorBox>;
}
