use mlframework::tensor::Tensor;

fn main() {
    let x = Tensor::new(vec![3, 4, 5]);
    let y = Tensor::new(vec![1, -2, 1]);
    let z = Tensor::new(vec![-3, 1, 3]);
    let s = (x + y.clone() + y) * z.relu();
    println!("s = {:?}", s);
    println!("s.traversal_ordering() {:?}", s.traversal_ordering());
    println!("s.id_to_tensor_ref() {:?}", s.build_graph());

    let graph = s.build_graph();
    println!("graph {:?}", graph);

    let ordering_map = s.traversal_ordering();
    let mut ordering_vec: Vec<_> = ordering_map.iter().collect();
    ordering_vec.sort_by(|a, b| a.1.cmp(b.1));
    println!("ordering_vec {:?}", ordering_vec);
}
