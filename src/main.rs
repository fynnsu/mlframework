#![feature(generic_const_exprs)]
use mlframework::{
    ops::Op,
    reshape::{Flattens, Reshapes},
    shape::{Const, D1, D2, D3},
    tensor::Tensor,
};

fn main() {
    let x: Tensor<f32, (Const<20>, Const<1>)> = Tensor::new([[0.0]; 20]);
    // let y = Tensor::new(vec![1., -2., 1.]);
    // let z = Tensor::new(vec![-3., 1., 3.]);

    // let s = ((x + y.clone() + y) * z.relu()).reduce_sum();

    let x_225: Tensor<f32, (Const<2>, Const<2>, Const<5>)> = x.reshape();
    let x_20: Tensor<f32, (Const<20>,)> = x_225.reshape();

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
