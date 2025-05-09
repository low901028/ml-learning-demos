use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::iter::Map;
use std::ops::{Add, Mul};
use candle_core::{Result, Tensor, Device, Module, DType, CustomOp1, Layout, Shape, StreamTensor, WithDType, D, IndexOp, pickle};
use candle_core::pickle::{read_pth_tensor_info, TensorInfo};
use candle_core::quantized::QMatMul::TensorF16;
use candle_datasets::vision::Dataset;
use candle_nn::ops::{sigmoid, softmax};
use clap::arg;
use clap::builder::Resettable::Reset;
use futures::StreamExt;
use image::{DynamicImage, GrayImage};
use plotters::prelude::{BitMapBackend, ChartBuilder, Color, IntoDrawingArea, IntoFont, LineSeries, PathElement, BLACK, RED, WHITE};
use serde::Deserialize;
use serde_pickle::DeOptions;
use tokio::task::id;

/// 感知机
///
///    | --- w1*x1 + w2*x2 + b <= 0
/// y =
///    | --- w1*x1 + w2*x2 + b > 0
///
/// w权重：代表输入信号的重要程度
/// b偏置：代表神经元被激活的容易程度
/// 根据电路门的需要设置不同的权重和偏置
fn naive_perceptron_demo(device: &Device) -> Result<()>{
    fn AND(x1: f32, x2: f32) -> u8{
        let (w1, w2, theta) = (0.5f32, 0.5, 0.7f32);
        let tmp = x1.mul(w1).add(x2.mul(w2));
        let res = if tmp.le(&theta) {
            0
        } else {
            1
        };

        res
    }

    fn NAND(x1: f32, x2: f32) -> u8{
        let (w1, w2, theta) = (-0.5f32, -0.5, -0.7f32);
        let tmp = x1.mul(w1).add(x2.mul(w2));
        let res = if tmp.le(&theta) {
            0
        } else {
            1
        };

        res
    }

    fn OR(x1: f32, x2: f32) -> u8{
        let (w1, w2, theta) = (0.5f32, 0.5, -0.0f32);
        let tmp = x1.mul(w1).add(x2.mul(w2));
        let res = if tmp.le(&theta) {
            0
        } else {
            1
        };

        res
    }

    // 0, 0, 0, 1
    println!("perceptron AND GATE demo: {:?}, {:?}, {:?}, {:?}"
             ,AND(0.,0.)
             ,AND(1.,0.)
             ,AND(0.,1.)
             ,AND(1.,1.)
    );

    //  1, 1, 1, 0
    println!("perceptron NAND GATE demo: {:?}, {:?}, {:?}, {:?}"
             ,NAND(0.,0.)
             ,NAND(1.,0.)
             ,NAND(0.,1.)
             ,NAND(1.,1.)
    );

    //  0, 1, 1, 1
    println!("perceptron OR GATE demo: {:?}, {:?}, {:?}, {:?}"
             ,OR(0.,0.)
             ,OR(1.,0.)
             ,OR(0.,1.)
             ,OR(1.,1.)
    );

    fn ndarray_demo(device: &Device) -> Result<()>{
        let x = Tensor::from_slice(&[0f32,1f32], (2,), device)?;
        let w = Tensor::from_slice(&[0.5f32, 0.5], (2,), device)?;
        let b = -0.7;
        println!("w*x={}", (w.clone()*x.clone())?);
        println!("sum(w*x)={}", (w.clone()*x.clone())?.sum_all()?);
        let result = ((w.clone()*x.clone())?.sum_all()? + b)?;
        println!("sum(w*x) + b ={}", result);
        Ok(())
    }
    //ndarray_demo(device)?;

    fn AND_Gate(x1:f32, x2:f32, device: &Device) -> Result<u8>{
        let x = Tensor::from_slice(&[x1,x2], (2,), device)?;
        let w = Tensor::from_slice(&[0.5f32,0.5f32], (2,), device)?;
        let b = -0.7;

        let res = if ((w*x)?.sum_all()? + b)?.to_scalar::<f32>()?.le(&0f32) {
            0
        } else {
            1
        };

        Ok(res)
    }

    // 0, 0, 0, 1
    println!("perceptron AND-GATE-V2 demo: {:?}, {:?}, {:?}, {:?}"
             ,AND_Gate(0.,0., device)?
             ,AND_Gate(1.,0., device)?
             ,AND_Gate(0.,1., device)?
             ,AND_Gate(1.,1., device)?
    );

    fn NAND_Gate(x1:f32, x2:f32, device: &Device) -> Result<u8>{
        let  x = Tensor::from_slice(&[x1,x2], (2,), device)?;
        let w = Tensor::from_slice(&[-0.5f32, -0.5f32], (2,), device)?;
        let b = 0.7;
        let tmp  = ( (w*x)?.sum_all()? + b )?;
        let res = if tmp.to_scalar::<f32>()?.le(&0f32) {
            0
        } else {
            1
        };

        Ok(res)
    }

    fn OR_Gate(x1:f32, x2:f32, device: &Device) -> Result<u8>{
        let x = Tensor::from_slice(&[x1,x2], (2,), device)?;
        let  w = Tensor::from_slice(&[0.5f32, 0.5f32], (2,), device)?;
        let b = -0.2;

        let tmp = ((w*x)?.sum_all()? + b)?;
        let res = if tmp.to_scalar::<f32>()?.le(&0f32) {
            0
        } else {
            1
        };

        Ok(res)
    }

    // 输出：1, 1, 1, 0
    println!("perceptron NAND-GATE-V2 demo: {:?}, {:?}, {:?}, {:?}"
             ,NAND_Gate(0.,0., device)?
             ,NAND_Gate(1.,0., device)?
             ,NAND_Gate(0.,1., device)?
             ,NAND_Gate(1.,1., device)?
    );
    // 输出：  0, 1, 1, 1
    println!("perceptron OR-GATE-V2 demo: {:?}, {:?}, {:?}, {:?}"
             ,OR_Gate(0.,0., device)?
             ,OR_Gate(1.,0., device)?
             ,OR_Gate(0.,1., device)?
             ,OR_Gate(1.,1., device)?
    );

    ///
    /// 通过与非门+或门来处理输入结果，再将两者的结果进行与门，最终得到异或门的结果
    /// 感知机并不能表示由一条直线分割的空间，故而需要通过引入多层感知机来解决线性与非线性的分割
    ///
    fn XOR_Gate(x1:f32, x2:f32, device: &Device) -> Result<u8>{
        /// 与非门
        let s1 = NAND_Gate(x1, x2, device)?;
        /// 或门
        let  s2 = OR_Gate(x1, x2, device)?;
        /// 与门
        let res = AND_Gate(s1 as f32, s2 as f32, device)?;
        Ok(res)
    }

    // 输出：  0, 1, 1, 0
    println!("perceptron XOR-GATE-V2 demo: {:?}, {:?}, {:?}, {:?}"
             ,XOR_Gate(0.,0., device)?
             ,XOR_Gate(1.,0., device)?
             ,XOR_Gate(0.,1., device)?
             ,XOR_Gate(1.,1., device)?
    );

    Ok(())
}


fn draw_preceptron(x: &Tensor, y: &Tensor) -> Result<()>{
    let root = BitMapBackend::new("plotters-preceptron_0.png", (640, 480))
        .into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .top_x_label_area_size(6)
        .right_y_label_area_size(6)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-6f32..6f32, -0.1f32..1.1f32).unwrap();

    chart.configure_mesh().disable_mesh().draw().unwrap();

   let points = x.to_vec1()?.into_iter().zip(y.to_vec1()?.into_iter()).map(|(x, y):(f32, f32)| {
        (x as f32 ,y  as f32)
    }).collect::<Vec<(f32, f32)>>();

    chart
        .draw_series(LineSeries::new(
            points.into_iter(),
            &BLACK,
        )).unwrap()
        ;
    root.present().unwrap();

    Ok(())
}
/// 多层网络
fn multiply_preceptron_demo(device: &Device) -> Result<()> {
    /// 自定义实现sigmoid
    fn sigmod(x: Tensor) -> Result<Tensor>{
       // 1.0 / (1.0 + Tensor::exp(&x.neg()?)?)?
       // (1.0 + Tensor::exp(&x.neg()?)?)?.recip()
       (&x.neg()?.exp()? + 1.0)?.recip()
    }
    let candle_sigmoid = candle_nn::Activation::Sigmoid;

    let x = Tensor::from_slice(&[1.0, 2.0], (2,), device)?;
    let sigmoid_x = sigmod(x.clone())?;
    println!("sigmoid_[1.0, 2.0]: {}", sigmoid_x);

    let sigmoid_x = candle_sigmoid.forward(&x.clone())?;
    println!("candle_nn_sigmoid_[1.0, 2.0]: {}", sigmoid_x);

    fn step_function(x: &Tensor, device: &Device) -> Result<Tensor> {
       // let res = if x.sum_all()?.to_scalar::<f32>()?.le(&0f32) {
       //      0
       //  } else {
       //      1
       //  };
       //
       //  Tensor::new(&[res as u8], device)
       let y =  x.gt(0.0)?.to_dtype(DType::U32);
       y
    }

    let result = step_function(&Tensor::new(&[3.0], device)?.to_dtype(DType::F32)?, device)?;
    println!("step_function_[3.0]: {}", result);

    let result = step_function(&Tensor::new(&[-1.0, 1.0, 2.0], device)?.to_dtype(DType::F32)?, device)?;
    println!("step_function_[-1.0, 1.0, 2.0]: {}", result);

    let x =  Tensor::arange_step(-5.0, 5.0, 0.1, device)?.to_dtype(DType::F32)?;
    let y = step_function(&x, &device)?.to_dtype(DType::F32)?;
    draw_preceptron(&x, &y)?;

    let x = Tensor::new(&[-1.0, 1.0, 2.0], device)?;
    println!("user define sigmoid(x): {}", sigmod(x.clone())?);

    let x = Tensor::new(&[1.0, 2.0, 3.0], device)?;
    println!("(x+1.0)={}", (1.0 + x.clone())?);
    println!("(1.0/x)={}", x.clone().recip()?);

    let x = Tensor::arange_step(-5.0, 5.0, 0.1, device)?.to_dtype(DType::F32)?;
    let y = sigmod(x.clone())?;
    draw_preceptron(&x, &y)?;

    // Relu
    fn relu(x: Tensor) -> Result<Tensor> {
        x.maximum(0f32)
    }
    let candle_relu = candle_nn::Activation::Relu;

    let x  = Tensor::new(&[-1.0, 1.0, 2.0], device)?;
    let y = relu(x.clone())?;
    println!("relu_[-1.0, 1.0, 2.0]: {}", y);
    let candle_y = candle_relu.forward(&x.clone())?;
    println!("candle_nn_relu_[-1.0, 1.0, 2.0]: {}", candle_y);

    let x = Tensor::new(&[1u32, 2, 3, 4], device)?;
    println!("Tensor x[1u32, 2, 3, 4] dim={:?}, shape={:?}, shape_0={:?}", x.dims(), x.shape(), x.shape().dim(0));
    let x = Tensor::new(&[[1u32,2], [3,4], [5,6]], device)?;
    println!("Tensor x[[1u32,2], [3,4], [5,6]] dim={:?}, shape={:?}, shape_0={:?}", x.dims(), x.shape(), x.shape().dim(0));

    let A = Tensor::new(&[[1u32, 2], [3, 4]], device)?;
    let B = Tensor::new(&[[5u32,6], [7, 8]], device)?;
    let dot = Tensor::mul(&A, &B)?;
    println!("A dot B={}", dot);

    let A = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], device)?;
    let B = Tensor::new(&[[5f32, 6.]], device)?;
    let dot = Tensor::matmul(&A, &B.t()?)?;
    println!("A dot B={}", dot);

    let X = Tensor::new(&[[1f32, 2.]], device)?;
    let W = Tensor::new(&[[1f32, 3., 5.], [2., 4., 6.]], device)?;
    let dot = Tensor::matmul(&X, &W)?;
    println!("X dot W={}", dot);

    let x = Tensor::new(&[[1f32,  0.5]], device)?;
    let w1 = Tensor::new(&[[0.1f32, 0.3, 0.5], [0.2, 0.4, 0.6]], device)?;
    let b1 = Tensor::new(&[[0.1f32, 0.2, 0.3]], device)?;
    let a1 = Tensor::matmul(&x, &w1)?.add(&b1)?;
    let z1 = sigmoid(&a1.clone())?;
    println!("x dot w1 add b1: {}, z1={}", a1, z1);

    let w2 = Tensor::new(&[[0.1f32, 0.4], [0.2, 0.5], [0.3, 0.6]], device)?;
    let b2 = Tensor::new(&[[0.1f32, 0.2]], device)?;
    let  a2 = Tensor::matmul(&a1, &w2)?.add(&b2)?;
    let z2 = sigmoid(&a2.clone())?;
    println!("x dot w2 add b2: {}, z2={}", a2, z2);

    fn identity_function(x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }

    let w3 = Tensor::new(&[[0.1f32, 0.3], [0.2, 0.4]], device)?;
    let b3 = Tensor::new(&[[0.1f32,0.2]], device)?;
    let a3 = Tensor::matmul(&z2, &w3)?.add(&b3)?;
    let z3 = identity_function(&a3)?;
    println!("x dot w3 add b3: {}, z3={}", a3, z3);

    struct NetWork {
        network: HashMap<&'static str, Tensor>,
    }
    fn init_network(device: &Device) -> Result<NetWork>{
        let mut network = NetWork {
            network: HashMap::with_capacity(16),
        };
        let w1 = Tensor::new(&[[0.1f32, 0.3, 0.5], [0.2, 0.4, 0.6]], device)?;
        let b1 = Tensor::new(&[[0.1f32, 0.2, 0.3]], device)?;
        network.network.insert("w1", w1);
        network.network.insert("b1", b1);

        let w2 = Tensor::new(&[[0.1f32, 0.4], [0.2, 0.5], [0.3, 0.6]], device)?;
        let b2 = Tensor::new(&[[0.1f32, 0.2]], device)?;
        network.network.insert("w2", w2);
        network.network.insert("b2", b2);

        let w3 = Tensor::new(&[[0.1f32, 0.3],[0.2, 0.4]], device)?;
        let  b3 = Tensor::new(&[[0.1f32, 0.2]], device)?;
        network.network.insert("w3", w3);
        network.network.insert("b3", b3);

        Ok(network)
    }

    fn forward(net_work: NetWork, x: &Tensor, device: &Device) -> Result<Tensor> {
        let (w1, w2, w3) = (net_work.network.get("w1").unwrap()
                            , net_work.network.get("w2").unwrap()
                            , net_work.network.get("w3").unwrap() );

        let (b1, b2, b3) = (net_work.network.get("b1").unwrap()
                            , net_work.network.get("b2").unwrap()
                            , net_work.network.get("b3").unwrap() );

        let a1 = Tensor::matmul(x, w1)?.add(b1)?;
        let z1 = sigmoid(&a1.clone())?;

        let a2 = Tensor::matmul(&z1, w2)?.add(b2)?;
        let z2 = sigmoid(&a2.clone())?;

        let a3 = Tensor::matmul(&z2, w3)?.add(b3)?;
        let y = identity_function(&a3)?;

        Ok(y.clone())
    }

    let network = init_network(&device)?;
    let x = Tensor::new(&[[1.0f32, 0.5]], device)?;
    let y = forward(network, &x, &device)?;
    println!("y={:.}", y);

    ///
    /// exp(x-c) / sum(exp(x-c))
    /// 因exp函数单调递增，故而存在“溢出”的问题，输出：INF
    /// 通过减去输入集中最大的元素，降低溢出风险
    ///
    fn softmax(x: Tensor) -> Result<Tensor> {
        let c = x.max_all()?;
        let exp_a = x.clone().broadcast_sub(&c.clone())?.exp()?;
        let sum_exp_a= exp_a.clone().sum_all()?;
        let y = exp_a.clone().broadcast_div(&sum_exp_a.clone())?;

        Ok(y)
    }

    let candle_softmax = candle_nn::ops::softmax;
    let x = Tensor::new(&[[0.3f32, 2.9, 4.0]], device)?;
    let exp_a = Tensor::exp(&x.clone())?;
    println!("exp(x)={}", exp_a.clone());
    let sum_exp_a = Tensor::sum_all(&exp_a.clone())?;
    println!("sum_exp(exp(x))={:?}", sum_exp_a);
    let y = exp_a.clone().broadcast_div(&sum_exp_a)?;
    let softmax_y = softmax(x.clone())?;
    let candle_softmax_y = candle_softmax(&x.clone(), 1)?;
    println!("softmax: y={}, softmax_y={}, candle_softmax_y={}", y, softmax_y, candle_softmax_y);

    let a = Tensor::new(&[[1010f32, 1000., 900.]], device)?;
    /// 如下代码在未进行减去max(x)
    ///  - softmax = exp(x)/sum(exp(x))
    let inf_softmax =  softmax(a.clone())?;
    println!("inf(x)={}", inf_softmax.clone());

    let a = Tensor::new(&[[0.3f32, 2.9, 4.]], device)?;
    let y =  softmax(a.clone())?;
    println!("y={:.}", y);

    Ok(())
}

fn load_fashion_mnist() -> Result<Dataset> {
    let home_dir = std::path::Path::new("datas");
    let dataset = candle_datasets::vision::mnist::load_dir(home_dir)?;
    println!(
        "train images shape={:?}; labels={:?}",
        dataset.train_images.shape(),
        dataset.train_labels.shape()
    );
    println!(
        "test images shape={:?}; labels={:?}",
        dataset.test_images.shape(),
        dataset.test_labels.shape()
    );
    Ok(dataset)
}

fn show_image(img: Tensor, label: u32) -> Result<()> {
    let datas = img.to_vec2::<u8>()?;
    let bytes = datas.into_iter().flatten().collect::<Vec<u8>>();
    let grayimage = GrayImage::from_vec(28, 28,  bytes);
    grayimage.unwrap().save(format!("mnist_28_28_image_{:?}.png", label)).unwrap();
    Ok(())
}

fn load_weight_bais_from_pkl<T: AsRef<std::path::Path>>(dir: T, key: Option<&str>) -> Result<Tensor> {
    let path = dir.as_ref().join("sample_weight.pt");
    let data_tensor = candle_core::pickle::PthTensors::new(path, Some("state_dict"))?;
    let data = data_tensor.get(key.unwrap())?;
    let tenser = data.unwrap();
    Ok(tenser)
}

fn predict(network: &HashMap<&str, Tensor>, x: Tensor) -> Result<Tensor> {
    let (W1, W2, W3) = (network.get("W1").unwrap(),network.get("W2").unwrap(),network.get("W3").unwrap());
    let (b1, b2, b3) = (network.get("b1").unwrap(),network.get("b2").unwrap(),network.get("b3").unwrap());
 
    let a1 = Tensor::matmul(&x.clone(), &W1)?.broadcast_add(b1)?;
    let z1 = sigmoid(&a1.clone())?;

    let a2 = Tensor::matmul(&z1.clone(), &W2)?.broadcast_add(b2)?;
    let z2 = sigmoid(&a2.clone())?;

    let a3 = Tensor::matmul(&z2.clone(), &W3)?.broadcast_add(b3)?;
    // println!("a3={}", a3);
    let y = softmax(&a3.clone(), 1);
    
    y
}

#[derive(Debug, Deserialize)]
struct Params {
    W1: Vec<Vec<f32>>,
    W2: Vec<Vec<f32>>,
    W3: Vec<Vec<f32>>,

    b1: Vec<f32>,
    b2: Vec<f32>,
    b3: Vec<f32>,
}
fn read_params_from_file<P: AsRef<std::path::Path>>(path: P) -> std::result::Result<Params, Box<dyn std::error::Error>> {
    let buf_reader = BufReader::new(File::open(path)?);
    let u = serde_json::from_reader(buf_reader)?;
    // Return the `User`.
    Ok(u)
}
fn deep_learning_demo(device: &Device) -> Result<()> {
    let mnist_datasets = load_fashion_mnist()?;
    let train_images = mnist_datasets.train_images;
    let train_labels = mnist_datasets.train_labels;
    let train_labels = train_labels.to_device(device)?.to_dtype(DType::U32)?;

    let dir = "/Users/dalan/rustspace/ml-learning-demos/mnist/pkl";
    let mut network = HashMap::with_capacity(16);
    let path = "/Users/dalan/rustspace/ml-learning-demos/mnist/pkl/weight_bias.json";
    let data = read_params_from_file(path).unwrap();
    
    // /// 权重
    // let pickle_w1 = load_weight_bais_from_pkl(dir, Some("layer1.weight"))?;
    // let pickle_w2 = load_weight_bais_from_pkl(dir, Some("layer2.weight"))?;
    // let pickle_w3 = load_weight_bais_from_pkl(dir, Some("layer3.weight"))?;
    // /// 偏置
    // let pickle_b1 = load_weight_bais_from_pkl(dir, Some("layer1.bias"))?;
    // let pickle_b2 = load_weight_bais_from_pkl(dir, Some("layer2.bias"))?;
    // let pickle_b3 = load_weight_bais_from_pkl(dir, Some("layer3.bias"))?;
    let pickle_w1 = Tensor::from_vec(data.W1.into_iter().flatten().collect::<Vec<f32>>(), (784,50), device)?;
    let pickle_w2 = Tensor::from_vec(data.W2.into_iter().flatten().collect::<Vec<f32>>(), (50,100), device)?;
    let pickle_w3 = Tensor::from_vec(data.W3.into_iter().flatten().collect::<Vec<f32>>(), (100,10), device)?;
    let pickle_b1 = Tensor::from_vec(data.b1, (50,), device)?;
    let pickle_b2 = Tensor::from_vec(data.b2, (100,), device)?;
    let pickle_b3 = Tensor::from_vec(data.b3, (10,), device)?;
    // //
    network.insert("W1", pickle_w1.clone());
    network.insert("W2", pickle_w2.clone());
    network.insert("W3", pickle_w3.clone());
    network.insert("b1", pickle_b1.clone());
    network.insert("b2", pickle_b2.clone());
    network.insert("b3", pickle_b3.clone());
    
    // println!("pickle_w1={:?}", pickle_w1);
    /// 测试
    // let img = train_images.i(1)?;
    // let label = train_labels.i(1)?.to_scalar::<u32>()?;
    // println!("train images[0] = {}, label={:?}", img, label);
    // let img = img.reshape((28,28))?;
    // let img = (img * 255f64)?.to_dtype(DType::U8)?;
    // println!("convert to Tensor(u32)={}", img);
    // // let img = train_images.gather(&Tensor::new(&[[0u32]], device)?, 0)?;
    // // let label = train_labels.gather(&Tensor::new(&[0u32], device)?, 0)?;
    // // println!("train images[0] = {}, label={}", img, label);
    // show_image(img, label)?;

    let mut accuracy_cnt = 0u32;
    let len = train_images.dim(0)?;
    for i in  0..len{
        let img = train_images.i(i)?.unsqueeze(0)?//.reshape((1,784))?
            ;
        let label = train_labels.i(i)?;
        let y = predict(&network, img.clone())?;
        let argmax = y.clone().argmax(1)?;

        let sum = argmax.to_dtype(DType::U32)?.broadcast_eq(&label)?.sum_all()?.to_scalar::<u8>()?;
        accuracy_cnt += sum as u32;
        // let p = argmax.clone().to_vec1::<u32>()?.get(0).unwrap().clone();
        // if u32::eq(&p, &label)   {
        //     accuracy_cnt += 1;
        // }
    }
    println!("Accuracy: {:?}, accuracy_cnt={:?}", accuracy_cnt as f32 / len as f32, accuracy_cnt);
    
    let batch_size = 100;
    accuracy_cnt = 0u32;
    let batches = len / batch_size;
    for i in (0..batches) {
        let x_batch = train_images.narrow(0, i * batch_size, batch_size)?;
        let label_batch = train_labels.narrow(0, i * batch_size, batch_size)?.to_dtype(DType::U32)?;

        let y = predict(&network, x_batch.clone())?;
        let argmax = y.clone().argmax(1)?;
        let sum = argmax.to_dtype(DType::U32)?.eq(&label_batch)?.sum_all()?.to_scalar::<u8>()?;
        
        accuracy_cnt += sum as u32;
    }

    println!("[Batch] Accuracy: {:?}, accuracy_cnt={:?}", accuracy_cnt as f32 / len as f32, accuracy_cnt);
    
    Ok(())
}

fn main() {
    println!("deep learning get started...");
    let device = Device::cuda_if_available(0).unwrap();
    deep_learning_demo(&device).unwrap();
    // multiply_preceptron_demo(&device).unwrap();
    // naive_perceptron_demo(&device).unwrap();
}