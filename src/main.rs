#![feature(generic_const_exprs)]
use mlframework::tensor::Tensor;

fn main() {
    let x = Tensor::new([2.0; 3]);
    let x_cloned = x.clone();
    let y = Tensor::new([1., -2., 1.]);

    // let z = x + y;
    // println!("x + y = {:?}", z)
    let z = Tensor::new(vec![-3., 1., 3.]);
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
