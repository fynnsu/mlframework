use crate::{dtype::Dtype, shape::I, tensor::Tensor};

// Vec to constant sized tensor

impl<T: Dtype> From<Vec<T>> for Tensor<T, ()> {
    fn from(value: Vec<T>) -> Self {
        assert_eq!(value.len(), 0);
        unsafe { Self::from_vec_unchecked(value) }
    }
}
impl<const D1: usize, T: Dtype> From<Vec<T>> for Tensor<T, (I<D1>,)> {
    fn from(value: Vec<T>) -> Self {
        assert_eq!(value.len(), D1);
        unsafe { Self::from_vec_unchecked(value) }
    }
}
impl<const D1: usize, const D2: usize, T: Dtype> From<Vec<T>> for Tensor<T, (I<D1>, I<D2>)> {
    fn from(value: Vec<T>) -> Self {
        assert_eq!(value.len(), D1 * D2);
        unsafe { Self::from_vec_unchecked(value) }
    }
}

impl<const D1: usize, const D2: usize, const D3: usize, T: Dtype> From<Vec<T>>
    for Tensor<T, (I<D1>, I<D2>, I<D3>)>
{
    fn from(value: Vec<T>) -> Self {
        assert_eq!(value.len(), D1 * D2 * D3);
        unsafe { Self::from_vec_unchecked(value) }
    }
}

// Array to constant size tensor
impl<T: Dtype, const D1: usize> From<[T; D1]> for Tensor<T, (I<D1>,)> {
    fn from(value: [T; D1]) -> Self {
        unsafe { Self::from_vec_unchecked(value.into()) }
    }
}

impl<T: Dtype, const D1: usize, const D2: usize> From<[[T; D2]; D1]> for Tensor<T, (I<D1>, I<D2>)> {
    fn from(value: [[T; D2]; D1]) -> Self {
        unsafe { Self::from_vec_unchecked(value.concat()) }
    }
}

impl<T: Dtype, const D1: usize, const D2: usize, const D3: usize> From<[[[T; D3]; D2]; D1]>
    for Tensor<T, (I<D1>, I<D2>, I<D3>)>
{
    fn from(value: [[[T; D3]; D2]; D1]) -> Self {
        unsafe { Self::from_vec_unchecked(value.concat().concat()) }
    }
}
