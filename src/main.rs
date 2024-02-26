use mlframework::tensor::Tensor;

fn main() {
    let x = Tensor::new(vec![3, 4, 5]);
    let y = Tensor::new(vec![1, -2, 1]);
    let z = Tensor::new(vec![0, 1, 3]);
    let s = (x.clone() + y.clone()) * z.clone();
    println!("({:#?} + {:#?}) * {:#?} = {:#?}", x, y, z, s);
}
