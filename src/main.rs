#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use mlframework::{
    build_mod, change_dtype::Converts, reshape::Reshapes, s, t, tensor::remove_inputs, Tensor,
};

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

fn prepare_for_training() {
    let x = Tensor::new([[1.0; 3]; 4]);
    let x_clone = x.clone();
    let y = Tensor::new([[3.0; 7]; 4]);
    let y_clone = y.clone();
    let in_ids = [x.id, y.id];

    let w = Tensor::new_with_grad([[0.5; 7]; 3]);
    let y_hat = x.matmul(w);
    let diff = y - y_hat;
    let loss = (diff.clone() * diff.clone()).reduce_sum();

    let traced_model = Model::new(x_clone.clone(), y_clone, loss.clone());

    println!("Loss 1: {:?}", loss);
    traced_model.recompute(vec![1.3; 12], vec![3.1; 28]);
    // x_clone.replace_data_with(vec![-1.0; 12]);
    // loss.recompute();
    println!("Loss 2: {:?}", loss);

    let mut parameters = loss.leaves();
    remove_inputs(&mut parameters, &in_ids);
    loss.backward();

    for p in parameters {
        println!("Param({:?}, grad={})", p, p.tensor.grad_to_string());
    }
}

fn main() {
    let x = Tensor::new([2.0; 3]);
    let x_cloned = x.clone();
    let y = Tensor::new_with_grad([1., -2., 1.]);
    let z = Tensor::new_with_grad(vec![-3., 1., 3.]);
    let xxy = x + y.clone() + y;
    println!("xxy = {:?}", xxy);
    let zrelu = z.relu();
    println!("zrelu = {:?}", zrelu);
    let mul = xxy * zrelu;
    println!("mul = {:?}", mul);

    let s = mul.reduce_sum();
    println!("((x + y + y) * z.relu()).reduce_sum() = {:?}", s);

    s.backward();

    println!("\n x_cloned = {:?}", x_cloned);

    let s: Tensor<f64, _> = Tensor::new([2.0; 3]);
    let s2: Tensor<i32, _> = s.convert();
    println!("{:?}", s2);
    shapes();

    prepare_for_training();

    //0 x = (3, 4, 5)
    //1 y = (1, -2, 1)
    //2 z = (-3, 1, 3)
    //3 x + y = (4, 2, 6)
    //4 x + y + y = (5, 0, 7)
    //5 zeros_like(z) = (0, 0, 0)
    //6 z.relu() = z.max(zeros_like(z)) = (0, 1, 3)
    //7 s = (0, 0, 21)
    // s.reduce_sum() = (x + y + y) * z.relu()
    // println!("s = {:?}", s);
    // println!("s.traversal_ordering() {:?}", s.traversal_ordering());
    // println!("s.id_to_tensor_ref() {:?}", s.build_graph());

    // let graph = s.build_graph();
    // println!("graph {:?}", graph);

    // let ordering_map = s.traversal_ordering();
    // let mut ordering_vec: Vec<_> = ordering_map.iter().collect();
    // ordering_vec.sort_by(|a, b| a.1.cmp(b.1));
    // println!("ordering_vec {:?}", ordering_vec);

    // println!();
    // println!();
    // let grad = s.grad();
    // for (t_id, g) in grad.iter() {
    //     println!("Tensor {:?}, grad {:?}", graph.nodes.get(t_id), g)
    // }
    // println!("grad {:?}", grad);
}
