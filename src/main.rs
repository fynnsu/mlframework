#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use mlframework::{
    build_mod,
    change_dtype::Converts,
    optim::GradientDescent,
    random::randn,
    reshape::Reshapes,
    s, t,
    tensor::{remove_inputs, TensorTrait},
    Tensor,
};

fn main() {
    simple_computation();

    type_conversion();

    shapes();

    simple_training();
}

fn simple_computation() {
    let x = Tensor::new([2.0; 3]);
    let y = Tensor::new_with_grad([1., -2., 1.]);
    let y_cloned = y.clone();
    let z = Tensor::new_with_grad(vec![-3., 1., 3.]);
    let z_cloned = z.clone();

    let x_plus_yy = x + y.clone() + y;
    println!("x_plus_yy = {:?}", x_plus_yy);
    let zrelu = z.relu();
    println!("zrelu = {:?}", zrelu);
    let mul = x_plus_yy * zrelu;
    println!("mul = {:?}", mul);

    let s = mul.reduce_sum();
    println!("((x + y + y) * z.relu()).reduce_sum() = {:?}", s);

    s.backward();
    println!("z_grad = {:?}", z_cloned.grad_to_string());
}

fn type_conversion() {
    let s: Tensor<f64, _> = Tensor::new([2.0; 3]);
    let s2: Tensor<i32, _> = s.convert();
    println!("{:?}", s2);
}
fn shapes() {
    let x = Tensor::new([[4; 6]; 8]);
    let x: Tensor<_, s!(48)> = x.reshape();
    let x: Tensor<_, s!(2, 2, 12)> = x.reshape();
    let x: Tensor<_, s!(12, 4)> = x.reshape();
    let y = Tensor::new([[1; 7]; 4]);
    let m: t!(i32, (12, 7)) = x.matmul(y);
    println!("m = {:?}", m);
}

build_mod! {Model inputs=[x: t!(f64, (4, 3)), y: t!(f64, (4,7))], outputs=[loss: t!(f64, (1))]}

fn simple_training() {
    let x = Tensor::new([[1.0; 3]; 4]);
    let x_clone = x.clone();
    let y = Tensor::new([[3.0; 7]; 4]);
    let y_clone = y.clone();
    let in_ids = [x.id, y.id];

    let w = Tensor::new_with_grad([[0.5; 7]; 3]);
    let w_clone = w.clone();
    let y_hat = x.matmul(w);
    let diff = y - y_hat;
    let loss = (diff.clone() * diff.clone()).reduce_sum();

    let traced_model = Model::new(x_clone.clone(), y_clone, loss.clone());
    let mut opt = GradientDescent { lr: 0.01 };

    for i in 1..10 {
        traced_model.recompute(vec![1.3; 12], vec![3.1; 28]);
        println!("Loss {}: {:?}", i, loss);
        loss.backward();
        w_clone.consume_grad(&mut opt);
    }

    // x_clone.replace_data_with(vec![-1.0; 12]);
    // loss.recompute();

    let mut parameters = loss.leaves();
    remove_inputs(&mut parameters, &in_ids);
    loss.backward();

    for p in parameters {
        println!("Param({:?}, grad={})", p, p.tensor.grad_to_string());
    }

    let rng = rand::thread_rng();
    let r: Tensor<_, s!(3, 5, 1)> = randn(1.0, 1.0, rng);
    println!("Random Tensor {:?}", r);
}
