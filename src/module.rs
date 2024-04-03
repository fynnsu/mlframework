use crate::optim::Optimizer;

pub trait Module {
    type Input;
    type Output;
    fn forward(i: Self::Input) -> Self::Output;
    fn consume_grad<Opt: Optimizer>(optim: &mut Opt);
}
