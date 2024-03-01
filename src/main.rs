use mlframework::tensor::Tensor;

fn main() {
    let x = Tensor::new(vec![3, 4, 5]);
    let y = Tensor::new(vec![1, -2, 1]);
    let z = Tensor::new(vec![-3, 1, 3]);
    let s = (x + y) * z.relu();
    println!("s = {:?}", s);
    println!("s.traversal_ordering() {:?}", s.traversal_ordering());
    println!("s.id_to_tensor_ref() {:?}", s.id_to_tensor_ref());

    let id_to_tensor_ref = s.id_to_tensor_ref();

    let ordering_map = s.traversal_ordering();
    let mut ordering_vec: Vec<_> = ordering_map.iter().collect();
    ordering_vec.sort_by(|a, b| a.1.cmp(b.1));
    let t_vec: Vec<_> = ordering_vec
        .iter()
        .map(|v| id_to_tensor_ref.get(v.0))
        .collect();
    println!("ordering_vec {:?}", ordering_vec);
    println!("t_vec {:?}", t_vec);
}
