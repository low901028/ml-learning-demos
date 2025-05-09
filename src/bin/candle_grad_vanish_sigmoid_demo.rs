use candle_core::{Result, Device, Tensor, Module, Var};
use candle_core::quantized::QMatMul::TensorF16;
use ndarray::range;

///
/// sigmoid(x) = 1/(1+exp(-x))
/// sigmoid随着输入很大或很小时，则会出现梯度很小甚至消失
///
fn grad_vanish_sigmoid_demo(device: &Device) -> Result<()> {
    let x = Tensor::arange_step(-8.0, 8.0, 0.1, device)?;
    //
    let var_x = Var::from_tensor(&x)?;
    let y = candle_nn::activation::Activation::Sigmoid.forward(&var_x)?;
    let grad_store = y.backward()?;
    let grad = grad_store.get(&var_x).expect("no gradient");
    println!("source tensor:{}, grad={}", var_x, grad);

    Ok(())
}

/// 
/// 矩阵乘积发生“爆炸”
/// 
fn grad_explod_demo(device: &Device) -> Result<()> {
    let M = Tensor::randn(0.0f32, 1., (4,4), device)?;
    let mut Var_M = Var::from_tensor(&M)?;
    println!("source tensor:{}", Var_M);
    for _ in (0..100) {
        Var_M.set(&Var_M.matmul(&Var_M).unwrap());
    }

    println!("tensor matmul 100 times:{}", Var_M);

    Ok(())
}

fn main() {
    let device = Device::Cpu;
    grad_explod_demo(&device).unwrap();
    // grad_vanish_sigmoid_demo(&device).unwrap();
}