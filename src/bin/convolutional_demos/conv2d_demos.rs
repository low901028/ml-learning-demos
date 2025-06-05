use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use candle_core::{Result, Device, Tensor, DType, Var, IndexOp, TensorId};
use candle_nn::{*};
use candle_transformers::models::stable_diffusion::uni_pc::FinalSigmasType::SigmaMin;

/// 
/// 二维卷积
/// 
fn conrr2d(X: &Tensor, K: &Tensor, bias: Option<&Tensor>, device: &Device) -> Result<Tensor> {
    let (h, w) = (K.clone().shape().dims()[0], K.clone().shape().dims()[1]);
    let Y = Var::from_tensor(&Tensor::zeros((X.clone().shape().dims()[0] -h + 1, X.clone().shape().dims()[1] -w + 1), DType::F32, device)?)?;
    
    // Sum(Sub_X * Kenel)
    for i in (0..Y.shape().dims()[0]) {
        for j in (0..Y.shape().dims()[1]) {
            let sub_x = X.clone().i((i..i+h, j..j+w))?;
            let mul_sum = (sub_x.mul(&K))?.sum_all()?.reshape((1,1))?;
            let y_2d = Y.slice_assign(&[i..i+1, j..j+1], &mul_sum)?;
            Y.set(&y_2d.clone())?;
        }
    }
    
    let data = match bias {
        Some(b) => {Y.clone().as_tensor().broadcast_add(&b)?},
        None => {Y.clone().into_inner()},
    };
    //println!("sum(sub_x * k)={}", Y.clone());
    Ok(data)
}

#[derive(Clone)]
struct SimpleConv2d {
    weights: Var,
    bias: Var,
}


impl SimpleConv2d {
    fn _init_(kernel_size: &[usize], device: Device) -> Result<SimpleConv2d>{
        let weights = Var::from_tensor(&Tensor::randn(0f32, 1., kernel_size, &device)?)?;
        let bias = Var::from_tensor(&Tensor::randn(0f32, 1., 1, &device)?)?;
        
        Ok(SimpleConv2d{weights, bias})
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weights = self.weights.clone();
        let bias = self.bias.clone();

        conrr2d(&x.clone(), weights.as_tensor(), Some(bias.as_tensor()), x.clone().device())
    }
}

fn deep_learning_conv2d_demo(device: &Device) -> Result<()> {
    let X = Tensor::new(&[[0f32,1.,2.], [3.,4.,5.], [6., 7., 8.]], device)?;
    let K = Tensor::new(&[[0f32,1.], [2.,3.]], device)?;
    conrr2d(&X.clone(), &K.clone(), None, device)?;
    
    let conv2d = SimpleConv2d::_init_(&[2,2], device.clone().to_owned())?;
    conv2d.forward(&X.clone())?;
    
    Ok(())
}

fn deep_learning_conv2d_demo_v2(device: &Device) -> Result<()> {
    let X = Tensor::ones((6,8), DType::F32, device)?;
    println!("X = {}, id={:?}", X.clone(), X.clone().id());
    let X = X.slice_assign(&[0..X.clone().dims()[0],2..6], &Tensor::zeros((X.clone().dims()[0],4), DType::F32, device)?)?;
    println!("update X = {}, id={:?}", X.clone(), X.clone().id());

    let K = Tensor::new(&[[1f32,-1.]],  device)?;
    let Y = conrr2d(&X.clone(), &K.clone(), None, device)?;
    println!("K id={:?}, Y ={}, id={:?}", K.clone().id(), Y.clone(), Y.clone().id());

    let conv2d = SimpleConv2d::_init_(&[1,2], device.clone().to_owned())?;
    let step = 100;
    let lr = 3e-2;
    let X = X; //.reshape((1,1,6,8))?;
    let Y = Y.detach(); //.reshape((1,1,6,7))?;

    for i in 0..step {
        let y_hat = conv2d.forward(&X.clone())?;
        let l = ((y_hat - Y.clone())?.sqr()?).sum_all()?;
        let _gradstore = l.backward()?;
        let weights_grad = conv2d.weights.backward()?;
        let biass_grad = conv2d.bias.backward()?;
        {
            let weights = &conv2d.weights;
            let weight_grad = weights_grad.get_id(weights.id()).expect("weight grad...");
            let bias = &conv2d.bias;
            let bias_grad = biass_grad.get_id(bias.id()).expect("bias grad...");

            let new_weights = (weights.as_detached_tensor() - (weight_grad * lr)?)?;
            let _ = &conv2d.weights.set(&new_weights.clone())?;
            let new_bias = (bias.as_detached_tensor() - (bias_grad * lr)?)?;
            let _ = &conv2d.bias.set(&new_bias.clone())?;
        }
        

        // weights_grad.remove(&conv2d.weights);
        // biass_grad.remove(&conv2d.bias);
        println!("weight={}, bias={}", conv2d.weights, conv2d.bias);
        if ((i+1) % 5 == 0) {
            println!("step={}, loss={} ", (i+1), l.clone());
        }
    }
    println!("weight={}, bias={}", conv2d.weights, conv2d.bias);
    
    Ok(())
}

fn comp_conv2d(conv2d: impl Fn(&Tensor) -> Result<Tensor>, X: &Tensor) -> Result<Tensor> {
    let shape = X.shape().dims();
    // let (h, w) = (shape[0] + 1, shape[1] + 1);
    // let zeros = Tensor::zeros((h, w), X.clone().dtype(), X.clone().device())?;
    // let X = zeros.slice_assign(&[1..h,1..w], &X.clone())?;
    let X = X.clone().reshape((1,1, shape[0], shape[1]))?;
    let Y = conv2d(&X)?;

    Y.clone().squeeze(0)?.squeeze(0)
}

fn deep_learning_conv2d(device: &Device) -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, device);
    let conv_2d_cfg= Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    let conv2d = candle_nn::conv2d(1,1,3,conv_2d_cfg, vb)?;

    // 创建卷积层
    // let conv2d = Conv2d::new(
    //     Tensor::zeros((1, 1, 3, 3), DType::F32, &device)?, // 权重 (out_c, in_c, k, k)
    //     Some(Tensor::zeros(1, DType::F32, &device)?),       // 偏置
    //     conv_2d_cfg
    // );
    
    let X = Tensor::randn(0f32, 1., (8,8), device)?;
   
    let conv2d_fn = |x: &Tensor| -> Result<Tensor> {
        let x = &x.clone();
        conv2d.forward(&x)
    };
    
    let conv2d = comp_conv2d(conv2d_fn, &X.clone())?;
    let shape = conv2d.shape();
    println!("shape={:?}", shape);
    
    Ok(())
}

fn deep_learning_conv2d_v2(device: &Device) -> Result<()> {
    let config = Conv2dConfig {
        padding: 1, // 设置两侧各填充1行/列（总填充2行/列）
        ..Default::default() // 保留其他参数默认值（stride=1, dilation=1, groups=1）
    };

    // 创建卷积层
    let conv2d = Conv2d::new(
        Tensor::zeros((1, 1, 3, 3), DType::F32, &device)?, // 权重 (out_c, in_c, k, k)
        Some(Tensor::zeros(1, DType::F32, &device)?),       // 偏置
        config,
    );

    // 示例输入 (batch=1, channels=1, height=5, width=5)
    let input = Tensor::zeros((1, 1, 5, 5), DType::F32, &device)?;
    let output = conv2d.forward(&input)?;
    println!("Output shape: {:?}", output.shape());
    
    Ok(())
}

fn main() {
    let device = Device::cuda_if_available(0).unwrap();
    // deep_learning_conv2d_v2(&device).unwrap();
    deep_learning_conv2d(&device).unwrap();
    // deep_learning_conv2d_demo_v2(&device).unwrap();
    // deep_learning_conv2d_demo(&device).unwrap();
}