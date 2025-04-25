use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use ndarray::AssignElem;

///
fn candle_sequential_demo(device: &Device) -> Result<()> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let device_inner = device.clone();

    let linear = candle_nn::linear::linear(5 as usize, 5 as usize, vb.clone())?;
    
    let sequential = candle_nn::sequential::seq()
        .add(candle_nn::linear::linear(
            5 as usize,
            5 as usize,
            vb.clone(),
        )?)
        .add_fn(move |tensor| {
            let dims = tensor.dims();
            let init_ws = candle_nn::Init::Randn {
                mean: 0.,
                stdev: 1.,
            };
            vb.get_with_hints((dims[1], dims[0]), "weight", init_ws).unwrap();
            
            println!("====={}", tensor);
            let tensor = Tensor::randn(0f32, 0.01f32, tensor.shape(), &device_inner);
            tensor
        });

    let tensor = Tensor::rand(0f32, 25., (5, 5), &device.clone())?;
    println!("source tensor:{}", tensor);
    let tensor_linear = linear.forward(&tensor)?;
    println!("linear tensor:{}", tensor_linear);

    let tensor = sequential.forward(&tensor)?;
    println!("senquential tensor={}", tensor);

    Ok(())
}
fn main() {
    let device = Device::Cpu;
    candle_sequential_demo(&device).unwrap();
}
