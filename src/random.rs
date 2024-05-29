use rand::distributions::Distribution;
use rand::Rng;
use statrs::distribution::Normal;

use crate::{shape::Shape, Tensor};

pub fn randn<S: Shape, R: Rng>(mu: f64, sigma: f64, rng: R) -> Tensor<f64, S> {
    let normal = Normal::new(mu, sigma).unwrap();
    let v = normal.sample_iter(rng).take(S::NUM_ELS).collect();
    unsafe { Tensor::<f64, S>::from_vec_unchecked(v) }
}
