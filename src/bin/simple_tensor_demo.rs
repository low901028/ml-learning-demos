use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_core::pickle::TensorInfo;
use candle_core::quantized::QMatMul::TensorF16;
use candle_core::utils::cuda_is_available;
use safetensors::tensor;

fn tensor_shape(device: &Device) -> Result<()>{
    // shape(&[usize; C])
    let shape = [2,3,5];
    let tensor = Tensor::ones(&shape, DType::F32, device)?;
    println!("tensor_[2,] shape={:?}, dims={:?}, datas={}", tensor.shape(), tensor.shape().dims(), tensor);

    // shape(_,_,...)
    let tensor = Tensor::ones((2,), DType::F32, device)?;
    println!("tensor_() shape={:?}, dims={:?}, datas={}", tensor.shape(), tensor.shape().dims(), tensor);

    // shape(&[usize])
    let tensor = Tensor::ones(vec![2,3,5], DType::F32, device)?;
    println!("tensor_vec[] shape={:?}, dims={:?}, datas={}", tensor.shape(), tensor.shape().dims(), tensor);
    
    // shape(shape)
    let tensor = Tensor::ones(tensor.shape(), DType::F32, device)?;
    println!("tensor_shape(shape) shape={:?}, dims={:?}, datas={}", tensor.shape(), tensor.shape().dims(), tensor);

    // shape()
    let tensor = Tensor::ones((), DType::F32, device)?;
    println!("tensor_shape() shape={:?}, dims={:?}, datas={}", tensor.shape(), tensor.shape().dims(), tensor);

    // shape(3)
    let tensor = Tensor::ones(3, DType::F32, device)?;
    println!("tensor_shape(3) shape={:?}, dims={:?}, datas={}", tensor.shape(), tensor.shape().dims(), tensor);

    Ok(())
}

fn tensor_layout(device: &Device) -> Result<()> {
    let tensora = Tensor::ones((3,4), DType::F32, device)?;
    println!("tensor_layout: id={:?}, (layout) = {:?}, storage={:?}\n"
             , tensora.id(), tensora.layout(), tensora.storage_and_layout());
    
    // 输出的layout 可以看到stride=4，而不是0，说明底层storage是没有变的
    // 不过的对应的tensor id发生了变化
    // i()-> 在创建Tensor时，底层storage是直接clone源tensor的storage，故而两个tensor底层的storage是一样的
    let tensor = tensora.i(1)?;
    println!("tensor_i(1)_layout: id={:?}, (layout) = {:?}, storage={:?}\n"
             , tensor.id(), tensor.layout(), tensor.storage_and_layout());
    
    let indexes = Tensor::new(&[0u32,1,1,2], device)?.reshape((2,2))?;
    // 
    // 会将源tensor的内容read出来，提供给目标tensor，而不是直接对storage进行clone，
    // 故而源tensor和目标tensor的storage不一致
    // 
    let tensor = tensora.gather(&indexes, 0)?;
    println!("tensor_gather([[0,1],[1,2]])_layout: id={:?}, (layout) = {:?}, storage={:?}"
             , tensor.id(), tensor.layout(), tensor.storage_and_layout());
    
    Ok(())
}

fn main() {
    let device = Device::Cpu;
    tensor_layout(&device).unwrap();
    // tensor_shape(&device).unwrap();
}