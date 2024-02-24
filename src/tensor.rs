use std::{fmt, ops::Add};

#[derive(Debug)]
pub struct Tensor<T> {
    pub data: Vec<T>,
}

// impl<T: fmt::Display> fmt::Display for Tensor<T> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "Tensor({})", self.data)
//     }
// }

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T: Add<Output = T> + Copy> Add for Tensor<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert!(self.data.len() == other.data.len());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
        }
    }
}
