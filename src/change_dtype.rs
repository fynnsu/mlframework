use std::{marker::PhantomData, rc::Rc};

use crate::{
    dtype::Dtype,
    ops::{vec::el_unary, Op},
    shape::Shape,
    tensor::{Tensor, TensorBox, TensorTrait},
    tensor_data::TensorData,
};

use num::NumCast;

pub trait Converts<T: Dtype, S: Shape> {
    fn convert(self) -> Tensor<T, S>;
}

#[derive(Debug)]
pub struct ConvertStruct<T: Dtype, S: Shape, TensorType: Converts<T, S>> {
    data: TensorType,
    _dtype: PhantomData<T>,
    _shape: PhantomData<S>,
}

impl<T: Dtype, S: Shape, TensorType: Converts<T, S>> ConvertStruct<T, S, TensorType> {
    fn new(t: TensorType) -> Self {
        Self {
            data: t,
            _dtype: Default::default(),
            _shape: Default::default(),
        }
    }
}

impl<T: Dtype, S: Shape, OT: Dtype> Op for ConvertStruct<T, S, Tensor<OT, S>>
where
    Tensor<OT, S>: Converts<T, S>,
    T: NumCast,
    OT: NumCast,
{
    type Produces = Tensor<T, S>;

    fn propogate_grad(&self, t: &Self::Produces) {
        // t = change_dtype(a)
        if let Some(d_dt) = t.data.grad_ref().as_ref() {
            let d_da: Vec<OT> = el_unary(|v| NumCast::from(*v).unwrap(), d_dt);
            self.data.update_grad(d_da);
        } else {
            panic!("Attempted to propogate grad, but no grad value exists.")
        }
    }

    fn recompute(&self, t: &Self::Produces) {
        let data = el_unary(|v| NumCast::from(*v).unwrap(), &self.data.borrow_value());
        t.data.replace(data)
    }

    fn forward(self) -> Tensor<T, S> {
        let value = el_unary(|v| NumCast::from(*v).unwrap(), &self.data.borrow_value());
        let data = TensorData::new(value, self.data.requires_grad());
        unsafe { Self::Produces::from_rc_td_and_op_unchecked(data, Rc::new(self)) }
    }

    fn operands(&self) -> Vec<TensorBox> {
        vec![TensorBox::new(self.data.id, &self.data)]
    }
}

impl<T: Dtype, S: Shape, OT: Dtype> Converts<T, S> for Tensor<OT, S>
where
    T: NumCast,
    OT: NumCast,
{
    fn convert(self) -> Tensor<T, S> {
        // todo: This is probably too permission right now (especially since we just unwrap conversion failures)
        // Should probably impose stricter limitations on what can be converted. (Also backprop is a challenge)
        ConvertStruct::new(self).forward()
    }
}
