use std::{marker::PhantomData, rc::Rc};

use crate::{
    dtype::Dtype,
    ops::Op,
    shape::{HasNEls, Shape, D1, D2, D3, I},
    tensor::{Tensor, TensorBox},
};

pub trait Flattens<T: Dtype, const A: usize> {
    fn flatten(self) -> Tensor<T, (I<A>,)>;
}

#[derive(Debug)]
pub struct FlattenStruct<T: Dtype, const A: usize, TensorType: Flattens<T, A>> {
    data: TensorType,
    _dtype: PhantomData<T>,
}

impl<T: Dtype, const A: usize, TensorType: Flattens<T, A>> FlattenStruct<T, A, TensorType> {
    pub fn new(data: TensorType) -> Self {
        Self {
            data,
            _dtype: Default::default(),
        }
    }
}

impl<const A: usize, T: Dtype, S: Shape> Op for FlattenStruct<T, A, Tensor<T, S>>
where
    Tensor<T, S>: Flattens<T, A>,
{
    type Produces = Tensor<T, (I<A>,)>;

    fn propogate_grad(&self, _t: &Self::Produces) {
        // Flatten does not change data, therefore no grad propogation occurs
    }

    fn recompute(&self, _t: &Self::Produces) {}

    fn forward(self) -> Tensor<T, (I<A>,)> {
        unsafe {
            Self::Produces::from_rc_td_and_op_unchecked(self.data.data.clone(), Rc::new(self))
        }
    }

    fn operands(&self) -> Vec<TensorBox> {
        vec![TensorBox::new(self.data.id, &self.data)]
    }
}

impl<const A: usize, T: Dtype, S: Shape> Flattens<T, A> for Tensor<T, S>
where
    S: HasNEls<A>,
{
    fn flatten(self) -> Tensor<T, (I<A>,)> {
        FlattenStruct::new(self).forward()
    }
}

pub trait Reshapes<T: Dtype, S: Shape> {
    fn reshape(self) -> Tensor<T, S>;
}

#[derive(Debug)]
pub struct ReshapeStruct<T: Dtype, S: Shape, TensorType: Reshapes<T, S>> {
    data: TensorType,
    _dtype: PhantomData<T>,
    _shape: PhantomData<S>,
}

impl<T: Dtype, S: Shape, TensorType: Reshapes<T, S>> ReshapeStruct<T, S, TensorType> {
    pub fn new(data: TensorType) -> Self {
        Self {
            data,
            _dtype: Default::default(),
            _shape: Default::default(),
        }
    }
}

impl<T: Dtype, S: Shape, Si: Shape> Op for ReshapeStruct<T, S, Tensor<T, Si>>
where
    Tensor<T, Si>: Reshapes<T, S>,
{
    type Produces = Tensor<T, S>;

    fn propogate_grad(&self, _t: &Self::Produces) {
        // Reshape does not change data, therefore no grad propogation occurs
    }

    fn recompute(&self, _t: &Self::Produces) {}

    fn forward(self) -> Self::Produces {
        unsafe {
            Self::Produces::from_rc_td_and_op_unchecked(self.data.data.clone(), Rc::new(self))
        }
    }

    fn operands(&self) -> Vec<TensorBox> {
        vec![TensorBox::new(self.data.id, &self.data)]
    }
}

impl<const A: usize, T: Dtype, S: Shape> Reshapes<T, D1<A>> for Tensor<T, S>
where
    S: HasNEls<A>,
{
    fn reshape(self) -> Tensor<T, (I<A>,)> {
        ReshapeStruct::new(self).forward()
    }
}

impl<const A: usize, const B: usize, T: Dtype, S: Shape> Reshapes<T, D2<A, B>> for Tensor<T, S>
where
    S: HasNEls<{ A * B }>,
{
    fn reshape(self) -> Tensor<T, (I<A>, I<B>)> {
        ReshapeStruct::new(self).forward()
    }
}

impl<const A: usize, const B: usize, const C: usize, T: Dtype, S: Shape> Reshapes<T, D3<A, B, C>>
    for Tensor<T, S>
where
    S: HasNEls<{ A * B * C }>,
{
    fn reshape(self) -> Tensor<T, (I<A>, I<B>, I<C>)> {
        ReshapeStruct::new(self).forward()
    }
}
