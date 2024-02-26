use mlframework::tensor::Tensor;

fn main() {
    let x = Tensor::new(vec![3, 4, 5]);
    let y = Tensor::new(vec![1, -2, 1]);
    let s = x.clone() + y.clone();
    println!("{:#?} + {:#?} = {:#?}", x, y, s);
}
