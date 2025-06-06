use std::any::TypeId;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::iter::Map;
use std::ops::{Add, Deref, Div, Mul, Neg};
use candle_core::{Result, Tensor, Device, Module, DType, CustomOp1, Shape, StreamTensor, WithDType, D, IndexOp, pickle, Var};
use candle_core::op::Op;
use candle_core::scalar::TensorOrScalar;
use candle_datasets::vision::Dataset;
use candle_nn::ops::{sigmoid};
use candle_transformers::models::deepseek2::TopKLastDimOp;
use clap::arg;
use clap::builder::Resettable::Reset;
use clap::builder::TypedValueParser;
use futures::{FutureExt, StreamExt};
use image::{DynamicImage, GrayImage};
use plotters::prelude::{BitMapBackend, ChartBuilder, Color, IntoDrawingArea, IntoFont, LineSeries, PathElement, BLACK, RED, WHITE};
use rand::prelude::SliceRandom;
use serde::Deserialize;
use serde_pickle::DeOptions;
use tiff::tags::Tag::Software;
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

fn sigmod(x: Tensor) -> Result<Tensor>{
    // 1.0 / (1.0 + Tensor::exp(&x.neg()?)?)?
    // (1.0 + Tensor::exp(&x.neg()?)?)?.recip()
    (&x.neg()?.exp()? + 1.0)?.recip()
}

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
/// 多层网络
fn multiply_preceptron_demo(device: &Device) -> Result<()> {
    /// 自定义实现sigmoid

    let candle_sigmoid = candle_nn::Activation::Sigmoid;

    let x = Tensor::from_slice(&[1.0, 2.0], (2,), device)?;
    let sigmoid_x = sigmod(x.clone())?;
    println!("sigmoid_[1.0, 2.0]: {}", sigmoid_x);

    let sigmoid_x = candle_sigmoid.forward(&x.clone())?;
    println!("candle_nn_sigmoid_[1.0, 2.0]: {}", sigmoid_x);

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
    let y = candle_nn::ops::softmax(&a3.clone(), 1);

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


fn numerical_gradient(f: impl Fn(Tensor) -> Result<Tensor>, x: Tensor) -> Result<Tensor> {
    let h = 1e-4;

    let x_size = x.dims()[0];
    let mut datas: Vec<Tensor> = Vec::with_capacity(x_size);

    for idx in 0..x_size {
        let tmp_val = x.i(idx)?;
        // f(x+h)的计算
        let x_idx_add_h = (tmp_val.clone() + h.clone())?;
        let fxh1 = f(x_idx_add_h.clone())?;

        // f(x-h)的计算
        let x_idx_sub_h = (tmp_val.clone() - h.clone())?;
        let fxh2 = f(x_idx_sub_h.clone())?;

        let idx_grad = ((fxh1.clone() - fxh2.clone())? / (2f64*h))?;
        // println!("current_grad={}", idx_grad);
        datas.push(idx_grad.reshape((1,1))?);
    }

    Tensor::cat(&datas, 0)
}

fn cross_entropy_error(y: Tensor, t: Tensor) -> Result<Tensor> {
    let (y, t) = if y.dims().len() == 1 {
        (y.clone().reshape((1, y.clone().elem_count()))? , t.clone().reshape((1, t.clone().elem_count()))?)
    } else {
        (y, t.clone())
    };

    let batch_size = y.dims()[0];
    let delta = 1e-7;
    (t.to_dtype(DType::F32)? * ((y+delta)?.log()?))?.sum_all()?.neg()? / batch_size as f64
    // candle_nn::loss::cross_entropy(&y.unsqueeze(0)?.t()?, &t)
}

fn deep_learning_v2_demo(device: &Device) -> Result<()> {
    fn mean_squared_error(y: Tensor, t: Tensor) -> Result<Tensor> {
        Tensor::sum(&(y-t.to_dtype(DType::F32)?)?.sqr()?, 0)? / 2f64
        // (y-t)?.sqr()?.mean_all()
    }


    /**
    let t = Tensor::new(&[0u32,0,1,0,0,0,0,0,0,0], device)?.to_dtype(DType::U32)?;
    let y = Tensor::new(&[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], device)?.to_dtype(DType::F32)?;
    let mse = mean_squared_error(y.clone(), t.clone())?;
    let cross_entropy = cross_entropy_error(y.clone(), t.clone())?;
    println!("Mean squared error: {:?}", mse);
    println!("Cross entropy error: {:?}", cross_entropy);

    let y = Tensor::new(&[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0], device)?.to_dtype(DType::F32)?;
    let mse = mean_squared_error(y.clone(), t.clone())?;
    let cross_entropy = cross_entropy_error(y.clone(), t.clone())?;
    println!("Mean squared error: {:?}", mse);
    println!("Cross entropy error: {:?}", cross_entropy);
   */

    let mnist_datasets = load_fashion_mnist()?;
    let train_images = mnist_datasets.train_images;
    let train_labels = mnist_datasets.train_labels;
    let train_labels = train_labels.to_device(device)?.to_dtype(DType::U32)?;

    let test_images = mnist_datasets.test_images;
    let test_labels = mnist_datasets.test_labels;
    let test_labels = test_labels.to_device(device)?.to_dtype(DType::U32)?;
    println!("t_train_i_2={}", train_labels.gather(&Tensor::new(&[0u32,1], device)?, 0)?);

    /// 进行one-hot处理
    // depth 指定one-shot时 根据当前label的值将对应匹配位置置为1，其他位置为0
    // 比如将[9, 0] -> [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    //                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    let depth = mnist_datasets.labels;
    let on_value = 1;
    let off_value = 0;
    let train_labels = candle_nn::encoding::one_hot::<u32>(train_labels, depth, on_value, off_value)?;
    println!("x_train.shape={:?}, t_train.shape={:?};i_2={}", train_images.shape(), train_labels.shape(), train_labels.i(..2)?);


    let train_size = train_images.dim(0)? as u32;
    let batch_size = 10;
    
    /// 随机选择batch_size行记录
    let train_idxs = (0..train_size).collect::<Vec<_>>();
    let idxs = train_idxs
        .choose_multiple(&mut rand::thread_rng(), batch_size);
    let idxs_tensor = Tensor::from_iter(idxs.cloned().into_iter(),device)?
        .unsqueeze(1)?;
    println!("current train_idxs={}", &idxs_tensor);
    let x_batch = train_images.gather(&idxs_tensor.repeat((1, 784))?,  0)?;
    let t_batch = train_labels.gather(&idxs_tensor.repeat((1, 10))?,  0)?;
    println!("[batch] train data: image={}, label={}",  x_batch, t_batch);
    // 按批随机选择
    // let n_batches =  train_size / batch_size;
    // let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
    // batch_idxs.shuffle(&mut rand::thread_rng());
    // for batch_idx in batch_idxs.iter() {
    //     let batch_train_images = train_images.narrow(0, batch_size * batch_idx, batch_size)?;
    //     let batch_train_labels = train_labels.narrow(0, batch_size * batch_idx, batch_size)?;
    //
    //     println!("batch mask-train-images_shape:{:?}, label={:?}", batch_train_images.shape(), batch_train_labels.shape());
    // }
    
    /// 求导函数
    /// 求导公式与下面的实现存在差异
    /// df(x)/dx = [f(x+h)-f(x)]/h 存在在实现的时候，存在舍入误差，导致结果出现“偏差”
    /// 故而选择以x为中心，左右间距h，这样可以计算x左右差分误差
    /// 
    fn numerical_diff(f: impl Fn(Tensor) -> Result<Tensor>, x: Tensor) -> Result<Tensor> {
        let h = 1e-4; // 0.0001
        // f(x+h) -f(x-h) / (2*h)
        (f((x.clone()+h)?)? - f((x.clone()-h)?)?)? / (2f64*h)
    }
    
    /// 可以定义任意函数
    // y = 0.01x^2 + 0.1x
    fn function_1(x: Tensor) -> Result<Tensor> {
        (Tensor::sqr(&x)? * 0.01f64)? + (0.1f64 * x)?
    }

    let x = Tensor::arange_step(0.0f32, 20.0, 0.1, device)?;
    let y = function_1(x.clone())?;
    draw_preceptron(&x, &y)?;
    
    // 导数=0.01 * 2 * x + 0.1
    // 5->导数=0.2； 10->导数=0.3
    let local = Tensor::new(&[5u32, 10], device)?.to_dtype(DType::F32)?;
    let result  = numerical_diff(function_1, local.clone())?;
    println!("numerical_diff[5,10]={}", result);
    
    // f(x0, x1) = x0^2 + x1^2
    fn function_2(x: Tensor) -> Result<Tensor> {
        x.sqr()?.sum_all()
    }
    
    // x0=3, x1=4时，求x0的偏导数
    fn function_tmp1(x0: Tensor) -> Result<Tensor> {
        x0.sqr()? + 4.0f64 * 4.0
    }
    let datas = Tensor::new(&[3.0f32], device)?.to_dtype(DType::F32)?;
    println!("function_tmp1(x0)={}", numerical_diff(function_tmp1, datas.clone())?);
    
    // x0=3， x1=4的导数
    fn function_tmp2(x1: Tensor) -> Result<Tensor> {
        x1.sqr()? + 3.0f64 * 3.0
    }
    let datas = Tensor::new(&[4.0f32], device)?.to_dtype(DType::F32)?;
    println!("function_tmp1(x0)={}", numerical_diff(function_tmp2, datas.clone())?);

    let points = Tensor::new(&[3.0f32, 4.0, 0.0, 2.0, 3.0, 0.0], device)?.to_dtype(DType::F32)?;
    let result = numerical_gradient(function_2, points.clone().reshape(((),1))?)?;
    println!("function_tmp2(x0=3.0, x1=4.0)={}", result);

    /// 梯度下降简单实现
    fn gradient_descent(f: impl Fn(Tensor)-> Result<Tensor>, init_x: Tensor, lr: f64, step_num: i32) -> Result<Tensor> {
        let x = Var::from_tensor(&init_x.clone())?;

        let ff = Box::new(f);
        let xx = x.clone();

        /// 1、迭代变更grad的值
        /// 2、使用numerical_gradient进行梯度计算，得到当前的梯度gradient
        /// 3、结合learning rate来变更当前的gradient
        /// 4、将最新的gradient进行变更，并将最新梯度tensor输出
        /// 5、待到迭代完成获取所有迭代结果中最新的那个即为所需的梯度
        let mut grad = (0.. step_num).into_iter().map(move |idx| {
            let  grad = numerical_gradient(ff.as_ref(), xx.clone().as_tensor().clone()).unwrap();
            let tmp_x = (xx.as_tensor().clone() - (grad * lr).unwrap()).unwrap();
            xx.set(&tmp_x.clone()).unwrap();
            // println!("current grad: {}", tmp_x.clone());
            xx.as_detached_tensor()
        }).collect::<Vec<_>>();

        let grad_desc = grad.pop().unwrap().clone();
        Ok(grad_desc)
    }

    /// 梯度下降简单应用
    let init_x = Tensor::new(&[-3.0f32, 4.0], device)?.to_dtype(DType::F32)?;
    let grad_desc = gradient_descent(function_2, init_x.clone().reshape(((),1))?, 0.1, 100)?;
    println!("grad desc(lr=0.1, step=100) ={}", grad_desc);

    let grad_desc = gradient_descent(function_2, init_x.clone().reshape(((),1))?, 10.0, 100)?;
    println!("grad desc(lr=10.0, step=100) ={}", grad_desc);

    let grad_desc = gradient_descent(function_2, init_x.clone().reshape(((),1))?, 1e-10, 100)?;
    println!("grad desc(lr=1e-10, step=100) ={}", grad_desc);

    #[derive(Clone)]
    struct SimpleNet {
        W: Tensor,
    }

    impl SimpleNet {
        fn _init_(device: &Device) -> Self {
            SimpleNet {
                // W: Tensor::new(&[[0.47355232f32, 0.9977393, 0.84668094],[0.85557411, 0.03563661, 0.69422093]], device).unwrap(),
                W: Tensor::randn(0.0f32, 1.0, (2,3), device).unwrap(),
            }
        }

        fn predict(self, x: Tensor) -> Result<Tensor> {
            x.broadcast_matmul(&self.W)
        }

        fn loss(self, x: Tensor, t: Tensor) -> Result<Tensor> {
            let z = self.predict(x)?;

            /// ================================================
            /// canndle_nn中的softmax 和 自定义cross_entropy_error
            /// ================================================
            let y = softmax(z)?;
            let loss = cross_entropy_error(y, t);

            loss
        }
    }

    let net = SimpleNet::_init_(&device);
    println!("init weight:{}", net.W);

    let x = Tensor::new(&[[0.6,  0.9]], device)?.to_dtype(DType::F32)?;
    let p = net.clone().predict(x.clone())?;
    println!("predict weight:{}", p);
    let max_idx = p.argmax(1)?;
    println!("max_idx={}", max_idx);

    let depth = 3;
    let on_value = 1;
    let off_value = 0;
    let t = candle_nn::encoding::one_hot::<u32>(max_idx, depth, on_value, off_value)?;
    println!("one_hot_value={}", t);

    let t = Tensor::new(&[[0u32, 0, 1]],device )?;
    let loss = net.clone().loss(x.clone(), t.clone())?;
    println!("loss function={}", loss);

    let xx = x.clone();
    let tt = t.clone();
    let net_loss = net.clone();
    let f: Box<dyn Fn(Tensor)-> Result<Tensor> > = Box::new(|w| -> Result<Tensor> {
        let r = net_loss.clone().loss(xx.clone(), tt.clone())?;
        Ok(r)
    });

    let W = net.clone().W.reshape(((),1))?;
    let dW = numerical_gradient(f, W)?.reshape(net.W.shape())?;
    println!("dW_grad={}", dW);

    Ok(())
}

fn deep_learning_twolayer_net_demo(device: &Device) -> Result<()> {
    #[derive(Clone)]
    struct TwoLayerNet {
        params: HashMap<&'static str, Tensor>,
    }

    impl TwoLayerNet {
        fn _init_(input_size: usize,
                  hidden_size: usize,
                  output_size: usize,
                  weighted_init_std: f64,
                  device: &Device
        ) -> Result<TwoLayerNet> {
            let mut net = TwoLayerNet {
                params: HashMap::with_capacity(8),
            };

            /// 初始化weight 和 bias
            net.params.insert("W1",(Tensor::randn(0f32, 1f32, (input_size, hidden_size), device)? * weighted_init_std)?);
            net.params.insert("b1",Tensor::zeros(hidden_size, DType::F32, device)? );

            net.params.insert("W2",(Tensor::randn(0f32, 1f32, (hidden_size, output_size), device)? * weighted_init_std)?);
            net.params.insert("b2",Tensor::zeros(output_size, DType::F32, device)? );

            Ok(net)
        }

        fn predict(self, x: Tensor) -> Result<Tensor> {
            let (W1, W2) = (self.params["W1"].clone(), self.params["W2"].clone());
            let (b1, b2) = (self.params["b1"].clone(), self.params["b2"].clone());

            let a1 = x.clone().matmul(&W1)?.broadcast_add(&b1)?;
            let z1 = candle_nn::activation::Activation::Sigmoid.forward(&a1)?;
            let a2 = z1.matmul(&W2)?.broadcast_add(&b2)?;
            let y = candle_nn::ops::softmax(&a2, 1);

            y
        }

        fn loss(self, x: Tensor, t: Tensor) -> Result<Tensor> {
            let y = self.predict(x)?;
            // cross_entropy_error(y, t)
            candle_nn::loss::cross_entropy(&y, &t)
        }

        fn accuracy(self, x: Tensor, t: Tensor) -> Result<f64> {
            let  y = self.predict(x.clone())?;
            let y = y.argmax(1)?;
            let t = t.argmax(1)?;
            let accuracy = (y.broadcast_eq(&t.clone())?.sum_all()?.to_scalar::<u32>()? as f64) / (x.clone().dims()[0] as f64);

            Ok(accuracy)

        }

        fn numerical_gradient(self, x: Tensor, t: Tensor) -> Result<HashMap<&'static str, Tensor>> {
            let loss_W: Box<dyn Fn(Tensor)-> Result<Tensor> > = Box::new(|w| -> Result<Tensor> {
                let r = self.clone().loss(x.clone(), t.clone())?;
                Ok(r)
            });

            let mut grads: HashMap<&str, Tensor> = HashMap::new();
            grads.insert("W1", numerical_gradient(&loss_W, self.params["W1"].clone())?);
            grads.insert("b1", numerical_gradient(&loss_W, self.params["b1"].clone())?);
            grads.insert("W2", numerical_gradient(&loss_W, self.params["W2"].clone())?);
            grads.insert("b2", numerical_gradient(&loss_W, self.params["b2"].clone())?);

            Ok(grads)
        }

    }

    let net = TwoLayerNet::_init_(784, 100, 10, 0.01f64, device)?;
    println!("net parameters shape w1={}, b1={}, w2={}, b2={}",
             net.params["W1"], net.params["b1"],
             net.params["W2"], net.params["b2"],
    );

    let x = Tensor::randn(0f32, 1f32, (100, 784), device)?;
    let y = net.clone().predict(x.clone())?;
    println!("x={}, y={}", x, y);
    let t = Tensor::randn(0f32, 1f32, (100, 10), device)?.argmax(1)?.to_dtype(DType::U32)?;
    println!("target tensor={}", t.clone());
    let grad = net.clone().numerical_gradient(x.clone(), t.clone())?;
    println!("grad parameters shape w1={}, b1={}, w2={}, b2={}",
             grad["W1"], grad["b1"],
             grad["W2"], grad["b2"],
    );

    let mnist_datasets = load_fashion_mnist()?;
    let train_images = mnist_datasets.train_images;
    let train_labels = mnist_datasets.train_labels;
    let train_labels = train_labels.to_device(device)?.to_dtype(DType::U32)?;

    let test_images = mnist_datasets.test_images;
    let test_labels = mnist_datasets.test_labels;
    let test_labels = test_labels.to_device(device)?.to_dtype(DType::U32)?;
    let depth = mnist_datasets.labels;
    let on_value = 1;
    let off_value = 0;
    // let train_labels = candle_nn::encoding::one_hot::<u32>(train_labels, depth, on_value, off_value)?;

    let mut train_loss_list: Vec<_> = vec![];
    let iters_num = 10000;
    let train_size = train_images.dims()[0] as u32;
    let batch_size = 100;
    let learning_rate = 0.1;
    let net = TwoLayerNet::_init_(784, 50, 10, 0.01f64, device)?;

    /// 随机选择batch_size行记录
    let train_idxs = (0..train_size).collect::<Vec<_>>();
    let mut params  = net.clone().params;
    let inner_net = net.clone();
    let inner_net_tmp = Box::new(net.clone());

    let idxs = train_idxs
        .choose_multiple(&mut rand::thread_rng(), batch_size);
    let idxs_tensor = Tensor::from_iter(idxs.cloned().into_iter(),device).unwrap()
        .unsqueeze(1).unwrap();
    let x_batch = train_images.clone().gather(&idxs_tensor.repeat((1, 784)).unwrap(),  0).unwrap();
    let t_batch = train_labels.clone().gather(&idxs_tensor.repeat((1, 10)).unwrap(),  0).unwrap();

    let grad = inner_net.numerical_gradient(x_batch.clone(), t_batch.clone()).unwrap();

    println!("grad: {:?}", grad);


    params.insert("W1", (&params["W1"]-(learning_rate * &grad.clone()["W1"]).unwrap()).unwrap());
    let loss = inner_net_tmp.clone().loss(x_batch.clone(), t_batch.clone()).unwrap();
    train_loss_list.push(loss);

    params.insert("b1", (&params["b1"]-(learning_rate * &grad.clone()["b1"]).unwrap()).unwrap());
    let loss = inner_net_tmp.clone().loss(x_batch.clone(), t_batch.clone()).unwrap();
    train_loss_list.push(loss);

    params.insert("W2", (&params["W2"]-(learning_rate * &grad.clone()["W2"]).unwrap()).unwrap());
    let loss = inner_net_tmp.clone().loss(x_batch.clone(), t_batch.clone()).unwrap();
    train_loss_list.push(loss);

    params.insert("b2", (&params["b2"]-(learning_rate * &grad.clone()["b2"]).unwrap()).unwrap());
    let loss = inner_net_tmp.clone().loss(x_batch.clone(), t_batch.clone()).unwrap();
    train_loss_list.push(loss);

    // (0..iters_num).for_each(move|i|{
    //
    // })
        // .for_each(move |(x_batch, t_batch, grad)| {
    //     for key in ["W1", "b1", "W2", "b2"] {
    //         params.insert(key, (&params[key]-(learning_rate * &grad.clone()[key]).unwrap()).unwrap());
    //         let loss = inner_net_tmp.clone().loss(x_batch.clone(), t_batch.clone()).unwrap();
    //         train_loss_list.push(loss);
    //     }
    // })
    ;

    println!("train_loss_list: {:?}", train_loss_list);

    Ok(())
}

trait Model {
    fn forward(&mut self, x: Tensor) -> Result<Tensor>;

    fn backward(&mut self, dout: Tensor) -> Result<Tensor>;
}

struct Relu {
    mask: Option<Var>,
}

impl Relu {
    fn _init_() -> Result<Self> {
        let relu = Relu {
            mask: None,
        };

        Ok(relu)
    }
}
impl Model for Relu {
    fn forward(&mut self, x: Tensor) -> Result<Tensor> {
        let zeros = x.zeros_like()?;
        self.mask = Some(Var::from_tensor(&x.clone().ge(&zeros.clone())?.to_dtype(x.dtype())?)?);

        let out = x.clone();
        out.maximum(&zeros.clone())

        // let mask = x.clone().le(&zeros)?;
        // self.mask.set(&mask.to_dtype(x.dtype())?)?;
        //
        // let out = x.clone();
        // let target = out.clone().gather(&self.mask.to_dtype(DType::U32)?,1)?;
        // out.clone().slice_assign(&[..target.shape().dims()[0], ..target.shape().dims()[1]], &Tensor::zeros_like(&target.clone())? )?;
    }

    fn backward(&mut self, dout: Tensor) -> Result<Tensor> {
        // let idxs = self.mask.clone().detach();
        // dout.slice_assign(&[..idxs.clone().dims()[0],..idxs.clone().dims()[1]], &Tensor::zeros_like(&self.mask.clone())? )?;
        // let dx = dout;

        // let zeros = dout.zeros_like()?;
        let dout = dout.clone().mul(self.mask.clone().unwrap().detach())?;
        let dx = dout;

        Ok(dx.clone())
        // let zeros = dout.zeros_like()?;
        // dout.maximum(&zeros.clone())
    }
}

struct Sigmod{
    out: Option<Var>,
}

impl Sigmod {
    fn _init_() -> Result<Sigmod> {
        let sigmod = Sigmod {
            out: None,
        };

        Ok(sigmod)
    }
}

impl Model for Sigmod {
    fn forward(&mut self, x: Tensor) -> Result<Tensor> {
        let out = (x.neg()?.exp()? + 1.0f64)?.recip()?;
        self.out = Some(Var::from_tensor(&out)?);

        Ok(out)
    }

    fn backward(&mut self, dout: Tensor) -> Result<Tensor> {
        let dx = dout.mul(self.out.clone().unwrap().detach().neg()? + 1.0f64)?.mul(self.out.clone().unwrap().detach())?;

        Ok(dx)
    }
}

#[derive(Clone)]
struct Affine {
    W: Tensor,
    b: Tensor,
    x: Option<Var>,
    dW: Option<Var>,
    db: Option<Var>,
}

impl Affine {
    fn _init_(W: Tensor, b: Tensor) -> Result<Self> {
        let affine = Affine {
            W: W.clone(),
            b: b.clone(),
            x: None,
            dW: None,
            db: None,
        };

        Ok(affine)
    }
}
impl Model for Affine {
    fn forward(&mut self, x: Tensor) -> Result<Tensor> {
        self.x = Some(Var::from_tensor(&x.clone())?);
        x.matmul(&self.W)?.broadcast_add(&self.b)
    }

    fn backward(&mut self, dout: Tensor) -> Result<Tensor> {
        let dx = dout.clone().matmul(&self.W.t()?)?;
        self.dW = Some(Var::from_tensor(&self.clone().x.unwrap().t()?.clone().matmul(&dout.clone())?)?);
        self.db = Some(Var::from_tensor(&dout.clone().sum(0)?)?);

        Ok(dx.clone())
    }
}

/// ```rust
///   class SoftmaxWithLoss:
///     def __init__(self):
///         self.loss = None # 损失
///         self.y = None    # softmax的输出
///         self.t = None    # 监督数据(one-hot vector)
///
///     def forward(self, x, t):
///         self.t = t
///         self.y = softmax(x)
///         self.loss = cross_entropy_error(self.y, self.t)
///
///         return self.loss
///
///     def backward(self, dout=1):
///         batch_size = self.t.shape[0]
///         dx = (self.y - self.t) / batch_size
///
///         return dx
/// ```
///
#[derive(Clone)]
struct SoftmaxWithLoss {
    loss: Option<Var>,
    y: Option<Var>,
    t: Option<Var>,
}

impl SoftmaxWithLoss {
    fn _init() -> Result<SoftmaxWithLoss> {
        let softmaxwithloss = SoftmaxWithLoss {
            loss: None,
            y: None,
            t: None,
        };

        Ok(softmaxwithloss)
    }

    fn forward(&mut self, x: Tensor, t: Tensor) -> Result<Tensor> {
        self.t = Some(Var::from_tensor(&t.clone())?);
        self.y =  Some(Var::from_tensor(&softmax(x.clone())?)?);
        self.loss = Some(Var::from_tensor(&cross_entropy_error(self.y.clone().unwrap().detach(), self.t.clone().unwrap().detach())?)?);
        let loss = self.loss.clone().unwrap().detach();

        Ok(loss)
    }

    fn backward(&self, dout: Tensor) -> Result<Tensor> {
        let batch_size = self.t.clone().unwrap().shape().dims()[0];
        let dx = (&self.y.clone().unwrap().detach() - &self.t.clone().unwrap().detach().to_dtype(DType::F32)?)? / batch_size as f64;

        dx
    }

}


fn backward_derivate_and_learning(device: &Device) -> Result<()> {
    #[derive(Clone)]
    struct MulLayer {
        x: Tensor,
        y: Tensor,
    }

    impl MulLayer {
        fn _init_(x: Tensor, y: Tensor) -> Result<MulLayer> {
            let mullayer = MulLayer {
                x: x.clone(),
                y: y.clone(),
            };

            Ok(mullayer)
        }

        fn forward(&self) -> Result<Tensor> {
            self.x.matmul(&self.y)
        }

        fn backward(&self, dout: Tensor) -> Result<(Tensor, Tensor)> {
            let dx = dout.matmul(&self.y)?;
            let dy = dout.matmul(&self.x)?;

            Ok((dx, dy))
        }
    }

    let apple = Tensor::new(&[[100f32]], device)?;
    let apple_num = Tensor::new(&[[2u32]], device)?.to_dtype(DType::F32)?;
    let tax = Tensor::new(&[[1.1f32]], device)?;
    // 苹果价格
    let mul_apple_layer = MulLayer::_init_(apple.clone(), apple_num.clone())?;
    let apple_price = mul_apple_layer.forward()?;
    // tax price
    let mul_tax_layer = MulLayer::_init_(apple_price.clone(), tax.clone())?;
    let price = mul_tax_layer.forward()?;
    println!("price: {}", price);

    let dprice = Tensor::new(&[[1u32]], device)?.to_dtype(DType::F32)?;
    let (dapple_price, dtax) = mul_tax_layer.backward(dprice.clone())?;
    let (dapple, dapple_num) = mul_apple_layer.backward(dapple_price.clone())?;
    println!("dapple={}, dapple_num: {}, dtax={}", dapple, dapple_num, dtax);

    #[derive(Clone)]
    struct AddLayer {}
    impl AddLayer {
        fn  _init_() -> Result<AddLayer> {
            let add = AddLayer {};

            Ok(add)
        }

        fn forward(&self, x: Tensor, y: Tensor) -> Result<Tensor> {
            x.clone().add(y.clone())
        }

        fn  backward(&self, dout: Tensor) -> Result<(Tensor, Tensor)> {
            let dx = (dout.clone() * 1f64)?;
            let dy = (dout.clone() * 1f64)?;

            Ok((dx, dy))
        }
    }

    let apple = Tensor::new(&[[100f32]], device)?;
    let apple_num = Tensor::new(&[[2u32]], device)?.to_dtype(DType::F32)?;
    let orange = Tensor::new(&[[150f32]], device)?;
    let orange_num = Tensor::new(&[[3u32]], device)?.to_dtype(DType::F32)?;
    let tax = Tensor::new(&[[1.1f32]], device)?;

    /// forward 前向传播
    let mul_apple_layer = MulLayer::_init_(apple.clone(), apple_num.clone())?;
    let apple_price = mul_apple_layer.forward()?;

    let mul_orange_layer = MulLayer::_init_(orange.clone(), orange_num.clone())?;
    let orange_price =  mul_orange_layer.forward()?;

    let add_apple_orange_layer = AddLayer::_init_()?;
    let all_price = add_apple_orange_layer.forward(apple_price.clone(), orange_price.clone())?;

    let mul_tax_layer = MulLayer::_init_(all_price.clone(), tax.clone())?;
    let price = mul_tax_layer.forward()?;
    println!("price: {}", price.clone());

    /// backward
    let dprice = Tensor::new(&[[1f32]], device)?;
    let (dall_price, dtax) = mul_tax_layer.backward(dprice.clone())?;
    let (dapple_price, dorange_price) = add_apple_orange_layer.backward(dall_price.clone())?;
    let (dorange, dorange_num) = mul_orange_layer.backward(dorange_price.clone())?;
    let (dapple, dapple_num) = mul_apple_layer.backward(dapple_price.clone())?;
    println!("dapple={}, dapple_num={}, dorange={}, dorange_num={}, tax={}"
             , dapple, dapple_num, dorange, dorange_num, tax);

    let x = Tensor::new(&[[1.0f32, -0.5], [-2.0, 3.0]], device)?;
    let zeros = x.clone().zeros_like()?;
    let mask = x.clone().le(&zeros)?; //(x.clone().maximum(&zeros)? + x.clone().minimum(&zeros)?)?;
    println!("mask: {}", mask);
    println!("out maks={}", x.maximum(&zeros.clone())?);

    let mut relu = Relu::_init_()?;
    let out = &relu.forward(x.clone())?;
    println!("[Relu] mask:{}, out: {}", &relu.mask.clone().unwrap(), out);

    // let xx = Tensor::new(&[[1.0f32, -0.1], [0.3, 3.]], device)?;
    let dout = &relu.backward(out.clone())?;
    println!("dout: {}", dout);

    //
    println!("=========User Define Sigmoid==========");
    let mut sigmod = Sigmod::_init_()?;
    let forward = sigmod.forward(x.clone())?;
    println!("forward: {}, sigmod_out={}", forward, &sigmod.out.clone().unwrap().clone());
    let backward = &sigmod.backward(forward.clone())?;
    println!("backward: {}, sigmod_out={}", backward, &sigmod.out.clone().unwrap().clone());

    println!("=========User Define Affine/SoftMax==========");
    let X = Tensor::randn(0f32, 1f32, (2,), device)?;
    let W = Tensor::randn(0f32, 1f32, (2,3), device)?;
    let B = Tensor::randn(0f32, 1f32, (3,), device)?;

    println!("Tensor random shape= X:{:?}, W:{:?}, B:{:?}", X.clone().shape(), W.clone().shape(), B.clone().shape());
    let Y = ((X.unsqueeze(0)?.broadcast_matmul(&W))? + B.unsqueeze(0)?)?;
    println!("Y={}", Y);

    let dy = Tensor::new(&[[1f32, 2., 3.], [4.,5., 6.]], device)?;
    let db = dy.clone().sum(0)?;
    println!("db: {}, dim_1-sum={}", db, dy.clone().sum(1)?);

    println!("============Affine================");
    let mut afffine = Affine::_init_(W.clone(), B.unsqueeze(0)?)?;
    let affine_out = afffine.forward(X.unsqueeze(0)?)?;
    let affine_dout = afffine.backward(affine_out.clone())?;
    println!("[Affine] out={}, dout={}", affine_out.clone(), affine_dout.clone());

    Ok(())
}

struct TwoLayerNet {
    params: HashMap<&'static str, Tensor>,
    // layers: HashMap<&'static str, &'static dyn Model>,
    affine1: Affine,
    relu: Relu,
    affine2: Affine,
    lastLayer: SoftmaxWithLoss,
}
impl TwoLayerNet {
    fn  _init_(input_size: usize, hidden_size: usize, output_size: usize, weight_init_std: f64, device: &Device) -> Result<TwoLayerNet> {
        /// 权重W 偏置b
        let mut params = HashMap::with_capacity(8);
        params.insert("W1", (Tensor::randn(0f32, 1f32, (input_size, hidden_size), device)?  * weight_init_std)? );
        params.insert("b1", Tensor::zeros(hidden_size, DType::F32, device)?);

        params.insert("W2", (Tensor::randn(0f32, 1f32, (hidden_size, output_size), device)?  * weight_init_std)? );
        params.insert("b2", Tensor::zeros(output_size, DType::F32, device)?);

        let affine1 = Affine::_init_(params["W1"].clone(), params["b1"].clone())?;
        let relu = Relu::_init_()?;
        let affine2 = (Affine::_init_(params["W2"].clone(), params["b2"].clone())?);
        let lastLayer = SoftmaxWithLoss::_init()?;


        Ok(TwoLayerNet { params, affine1, relu, affine2, lastLayer })

    }

    fn predict(&mut self , x: Tensor) -> Result<Tensor> {
        self.affine2.forward(self.relu.forward(self.affine1.forward(x.clone())?)?)
    }

    fn loss(&mut self, x: Tensor, t: Tensor) -> Result<Tensor> {
        let y = self.predict(x.clone())?;
        self.lastLayer.forward(y.clone(), t.clone())
    }

    fn accuracy(&mut self, x: Tensor, t: Tensor) -> Result<Tensor> {
        let y = self.predict(x.clone())?;
        let y = y.argmax(1)?;
        let t = if t.dims().len() != 1 {
            t.argmax(1)?
        }else {
            t
        };

        let accuracy = (Tensor::sum_all(&y.clone().eq(&t.clone())?.to_dtype(DType::F64)?)? / (x.clone().dims()[0] as f64))?;

        Ok(accuracy)
    }

    fn numerical_gradient(&mut self, x: Tensor, t: Tensor) -> Result<HashMap<&'static str, Tensor>> {
        let loss = self.loss(x.clone(), t.clone())?;
        let loss_w = move |tensor| -> Result<Tensor> {Ok(loss.clone())};
        let mut grads: HashMap<&'static str, Tensor> = HashMap::with_capacity(8);
        grads.insert("W1", numerical_gradient(loss_w.clone(), self.params["W1"].detach())?);
        grads.insert("b1", numerical_gradient(loss_w.clone(), self.params["b1"].detach())?);
        grads.insert("W2", numerical_gradient(loss_w.clone(), self.params["W2"].detach())?);
        grads.insert("b2", numerical_gradient(loss_w.clone(), self.params["b2"].detach())?);

        Ok(grads)
    }

    fn gradient(&mut self, x: Tensor, t: Tensor) -> Result<HashMap<&'static str, Tensor>> {
        &self.loss(x.clone(), t.clone())?;
        let dout = Tensor::new(&[1f32], x.device())?;
        let dout = self.lastLayer.backward(dout)?;
        // let dout_var = Var::from_tensor(&dout)?;

        let affine1 = &mut self.affine1;
        let relu = &mut self.relu;
        let affine2 = &mut self.affine2;
        affine1.backward(relu.backward(affine2.backward(dout)?)?)?;

        let mut grads: HashMap<&'static str, Tensor> = HashMap::with_capacity(8);
        grads.insert("W1", affine1.dW.clone().unwrap().detach());
        grads.insert("b1", affine1.db.clone().unwrap().detach());
        grads.insert("W2", affine2.dW.clone().unwrap().detach());
        grads.insert("b2", affine2.db.clone().unwrap().detach());

        Ok(grads)
    }

}
fn two_layer_deep_learning_demo(device: &Device) -> Result<()> {
    let mnist_datasets = load_fashion_mnist()?;
    let train_images = mnist_datasets.train_images;
    let train_labels = mnist_datasets.train_labels;
    let train_labels = train_labels.to_device(device)?.to_dtype(DType::U32)?;

    let test_images = mnist_datasets.test_images;
    let test_labels = mnist_datasets.test_labels;
    let test_labels = test_labels.to_device(device)?.to_dtype(DType::U32)?;

    let depth = mnist_datasets.labels;
    let on_value = 1;
    let off_value = 0;
    let train_labels = candle_nn::encoding::one_hot::<u32>(train_labels, depth, on_value, off_value)?;

    let mut network = TwoLayerNet::_init_(784, 50, 10, 0.01f64, device)?;
    let x_batch  = train_images.i(..3)?;
    let t_batch = train_labels.i(..3)?;
    println!("train batch images: {}, {}", x_batch, t_batch);

    let grad_numberical = network.numerical_gradient(x_batch.clone(), t_batch.clone())?;
    let grad_backprop = network.gradient(x_batch.clone(), t_batch.clone())?;

    for (k, v) in grad_numberical {
        let lhs = grad_backprop.get(k).unwrap();
        let diff = if lhs.clone().dims().len() >= v.clone().dims().len() {
            let lhs = lhs.clone().topk(1)?.indices.squeeze(0)?;
            lhs.clone().broadcast_sub(&v.clone().to_dtype(lhs.dtype())?.reshape((v.dims()[0], 1))?)?.abs()?.mean_all()?
        } else {
            let v = v.topk(1)?.indices.squeeze(0)?;
            v.clone().to_dtype(lhs.dtype())?.broadcast_sub(&lhs.clone().reshape((lhs.dims()[0], 1))?)?.abs()?.mean_all()?
        };
        
        println!("grad: key={:?} -> diff={}", k, diff);
    }

    Ok(())
}

fn two_layer_deep_learning_demo_v2(device: &Device) -> Result<()> {
    let mnist_datasets = load_fashion_mnist()?;
    let train_images = mnist_datasets.train_images;
    let train_labels = mnist_datasets.train_labels;
    let train_labels = train_labels.to_device(device)?.to_dtype(DType::U32)?;

    let test_images = mnist_datasets.test_images;
    let test_labels = mnist_datasets.test_labels;
    let test_labels = test_labels.to_device(device)?.to_dtype(DType::U32)?;

    let depth = mnist_datasets.labels;
    let on_value = 1;
    let off_value = 0;
    let train_labels = candle_nn::encoding::one_hot::<u32>(train_labels, depth, on_value, off_value)?;

    let mut network = TwoLayerNet::_init_(784, 50, 10, 0.01f64, device)?;
    
    let iters_num = 10000;
    let train_size = train_images.dims()[0] as u32;
    let batch_size = 100;
    let learning_rate = 0.1;
    let mut train_loss_list: Vec<Tensor> = vec![];
    let mut train_acc_list: Vec<Tensor> =  vec![];
    let mut test_acc_list: Vec<Tensor> =  vec![];

    let iter_per_epoch = (train_size / batch_size as u32).max(1) as u32;
    /// 随机选择batch_size行记录
    let train_idxs = (0..train_size).collect::<Vec<_>>();

    // let mut optimizer = SimpleSGD::_init_(0.01);
    // let mut optimizer = SimpleMomentum::_init_(0.01, 0.9);
    let mut optimizer = SimpleAdaGrad::_init_(0.01);
    
    for i in (1..=iters_num) {
        let idxs = train_idxs
            .choose_multiple(&mut rand::thread_rng(), batch_size);
        let idxs_tensor = Tensor::from_iter(idxs.cloned().into_iter(),device)?
            // .unsqueeze(1)?
            ;
        // println!("train_idxs: {}", idxs_tensor);
        // let x_batch = train_images.clone().gather(&idxs_tensor,  0)?;
        // let t_batch = train_labels.clone().gather(&idxs_tensor,  0)?;
        let x_batch = train_images.clone().index_select(&idxs_tensor, 0)?;
        let t_batch = train_labels.clone().index_select(&idxs_tensor,  0)?;

        // 通过误差反向传播法求梯度
        let grad = network.gradient(x_batch.clone(), t_batch.clone())?;
        let mut params = network.params.iter().map(|(k, v)|{
            (k.clone(), Var::from_tensor(&v.clone()).unwrap())
        }).collect::<HashMap<&str,Var>>();
        optimizer.update(&mut params, &grad)?;

        // 更新
        // for key in vec!["W1", "b1", "W2", "b2"] {
        //     let val = (network.params.get(key).unwrap() - (grad.get(key).unwrap().clone() * learning_rate)?)? ;
        //     network.params.insert(key, val);
        // }

        let loss = network.loss(x_batch.clone(), t_batch.clone())?;
        train_loss_list.push(loss.clone());

        if i  % iter_per_epoch == 0 {
            let train_acc = network.accuracy(train_images.clone(), train_labels.clone())?;
            let test_acc = network.accuracy(test_images.clone(), test_labels.clone())?;
            train_acc_list.push(train_acc.clone());
            test_acc_list.push(test_acc.clone());
            println!("acc train={}, test={}", train_acc.clone(), test_acc.clone());
        }
    }
    
    Ok(())
}

struct SimpleSGD {
    lr: f64
}

impl SimpleSGD {
    fn _init_(lr: f64) -> Self {
        SimpleSGD { lr }
    }

    fn update(
        &mut self,
        params: &mut HashMap<&str, Var>,
        grads: &HashMap<&str, Tensor>, // 改为不可变借用
    ) -> Result<()> {
        // 1. 优化键集合处理
        let param_keys: Vec<&str> = params.keys().copied().collect();

        for key in param_keys { // 直接解构&str
            // 2. 安全获取梯度
            let grad = grads.get(&key).unwrap();

            // 3. 分步计算并优化错误处理
            let scaled_grad = (grad.detach() * self.lr)?;

            // 4. 使用可变引用直接修改参数
            let param = params.get_mut(&key)
            .expect("Parameter existence guaranteed by key collection");

            // 5. 避免临时变量克隆
            let updated_tensor = (param.as_tensor() - &scaled_grad)?;
            *param = Var::from_tensor(&updated_tensor)?;
        }

        Ok(())
    }

}

struct SimpleMomentum {
    lr: f64,
    momentum: f64,
    v: Option<HashMap<&'static str, Var>>,
}

impl SimpleMomentum {
    fn _init_(lr: f64, momentum: f64) -> Self {
        SimpleMomentum {
            lr,
            momentum,
            v:  None,
        }
    }

    fn update(
        &mut self,
        params: & mut HashMap<&str, Var>,
        grads: &HashMap<&str, Tensor>,
    ) -> Result<()> {
        // 第一阶段：提前获取需要的信息
        let params_keys: Vec<&str> = params.keys().copied().collect();
        let mut vv = self.v.clone();

        if vv.is_none() {
            let mut v_map = HashMap::new();

            for (key, val) in &mut *params {
                v_map.insert(*key, Var::from_tensor(&val.zeros_like()?)?);
            }

            vv = Some(v_map);
        }

        let vv = vv.clone().unwrap();
        // 第三阶段：参数更新
        for &key in &params_keys {
            // 安全获取梯度（独立作用域）
            let scaled_grad = {
                let grad = grads.get(key).unwrap();

                let v = vv.get(key).unwrap();
                ((v.clone().as_tensor()*self.momentum)? - (grad.detach() * self.lr)?)?
            };

            // 获取参数可变引用
            let param = &params.get(key)
                .expect("Guaranteed by params_keys");

            // 执行更新
            let new_val = (param.clone().as_tensor() - &scaled_grad).unwrap();
            param.clone().set(&new_val)?;
        }

        Ok(())
    }
}

struct SimpleAdaGrad{
    lr: f64,
    h: Option<HashMap<&'static str, Var>>,
}

impl SimpleAdaGrad {
    fn _init_(lr: f64) -> SimpleAdaGrad {
        SimpleAdaGrad { lr, h: None }
    }

    fn update(
        &mut self,
        params: &mut HashMap<&str, Var>,
        grads: &HashMap<&str, Tensor>,
    ) -> Result<()>{
        let  param_keys: Vec<&str> = params.keys().copied().collect();

        let mut vv = self.h.clone();

        if vv.is_none() {
            let mut v_map = HashMap::new();

            for (key, val) in &mut *params {
                v_map.insert(*key, Var::from_tensor(&val.zeros_like()?)?);
            }

            vv = Some(v_map);
        }

        let vv = vv.clone().unwrap();
        for  key in param_keys {
            let val =  (vv.get(key).unwrap().as_tensor() - grads.get(key).unwrap().sqr()?)?;
            vv.get(&key).unwrap().set(&val)?;

            let new_val = (params.get(&key).unwrap().as_tensor() - ((grads.get(key).unwrap().detach() * self.lr)? / (vv.get(key).unwrap().sqrt()? + 1e-7)?)?)?;
            params.clone().get(&key).unwrap().set(&new_val)?
        }


        Ok(())
    }
}

fn one_hot_cold_demo(device: &Device) -> Result<()>{
    let indices = Tensor::new(vec![vec![0i64, 2], vec![1, -1]], device).unwrap();
    let depth = 4;
    let on_value = 1;
    let off_value = 0;
    let one_hot = candle_nn::encoding::one_hot::<u32>(indices, depth, on_value, off_value)?;
    let one_squeeze = one_hot.clone().topk(1)?.indices.squeeze(0)?;
    println!("one_hot_cold_demo: {}, one_squeeze={}", one_hot.clone(), one_squeeze);

    Ok(())
}

use plotly::common::{Font};
use plotly::layout::{Annotation, LayoutGrid, Layout, GridPattern};
use plotly::{Histogram, Plot};
use plotly::histogram::Bins;

fn plot_activations(activations: &HashMap<String, Vec<Vec<f32>>>) {
    let layer_count = activations.len();
    let mut plot = Plot::new();

    // 创建 Grid 布局
    let grid = LayoutGrid::new()
        .rows(1)
        .columns(layer_count)
        .pattern(GridPattern::Independent);

    // 收集所有直方图 traces
    let mut traces = Vec::new();
    let mut annotations = Vec::new();

    // 计算每个子图的 domain 宽度
    let domain_width = 1.0 / layer_count as f64;

    for (idx, (layer_name, a)) in activations.iter().enumerate() {
        // 扁平化数据
        let x_data: Vec<f32> = a.iter()
            .flatten()
            .cloned()
            .collect();

        // 创建直方图
        let trace = Histogram::new(x_data)
            .name(layer_name)
            .opacity(0.75)
            .x_bins(Bins::new(0.0, 1.0, 1.0/30.0));

        // 计算子图位置
        let x_domain_start = idx as f64 * domain_width;
        let x_domain_end = (idx as f64 + 1.0) * domain_width;

        // 设置坐标轴
        let x_axis_name = format!("x{}", idx + 1);
        let y_axis_name = format!("y{}", idx + 1);

        traces.push(
            trace
                .x_axis(&x_axis_name)
                .y_axis(&y_axis_name)
        );

        // 添加子图标题注解
        let annotation = Annotation::new()
            .x((x_domain_start + x_domain_end) / 2.0)
            .y(1.05)
            .ax_ref("paper")
            .ay_ref("paper")
            .text(format!("{}-layer", idx + 1))
            .show_arrow(false)
            .font(Font::new().size(12));

        annotations.push(annotation);
    }

    // 配置布局
    let layout = Layout::new()
        .grid(grid)
        .annotations(annotations)
        .show_legend(false)
        .height(400)
        .width(300 * layer_count);

    plot.set_layout(layout);

    // 添加 traces 到 plot
    for trace in traces {
        plot.add_trace(trace);
    }

    // 显示图表
    plot.show();
}

// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
fn simple_tanh(x: Tensor) -> Result<Tensor> {
    (x.clone().exp()? - x.clone().neg()?.exp()?)? / (x.clone().exp()? + x.clone().neg()?.exp()?)?
}

fn simple_relu(x: Tensor) -> Result<Tensor> {
    x.clone().maximum(0.0)
}
fn weight_decry_demo(device: &Device) -> Result<()>{
    let x = Tensor::randn(0f32, 1f32, (1000, 100), device)?;
    let node_num = 100;
    let hidden_layer_size = 5;
    let mut activations: Vec<Tensor>  = vec![];

    for i in (0..hidden_layer_size) {
        let x = if i != 0 {
            &activations[i - 1]
        } else {
            &x.clone()
        };

        let w = Tensor::randn(0f32, 1f32, (node_num, node_num), device)? * 1.0;

        // 会导致各层激活函数，集中在0.5左右，进而梯度表现力受限
        let w = Tensor::randn(0f32, 1f32, (node_num, node_num), device)? * 0.01;

        // xavier初始化参数的方式来
        // 各层间传递的数据有适当的广度
        let w = Tensor::randn(0f32, 1f32, (node_num, node_num), device)? / (node_num.isqrt() as f64);

        // he初始化参数，且激活函数为Relu
        let w = Tensor::randn(0f32, 1f32, (node_num, node_num), device)? / (2.0.div(node_num as f64).sqrt());
        let z = x.matmul(&w?)?;
        // let a = sigmod(z.clone())?;
        // let a = simple_tanh(z.clone())?;
        let a = simple_relu(z.clone())?;
        activations.insert(i, a.clone())
    }

    use plotly::{Histogram,};
    use plotly::layout::{GridPattern, Layout, LayoutGrid,};

    let mut datas = HashMap::new();
    for (i, t) in activations.iter().enumerate() {
        let data = t.clone().to_vec2::<f32>()?;
        datas.insert(format!("{:?}-layer",(i+1)), data);
    }

    plot_activations(&datas);

    Ok(())
}

struct SimpleDropout {
    dropout_ratio: f64,
    mask: Option<Var>,
}

impl SimpleDropout {
    fn _init_(dropout_ratio: f64) -> SimpleDropout {
        SimpleDropout {
            dropout_ratio,
            mask: None,
        }
    }
    
    fn forward(&mut self, x: Tensor, train_flag: bool) -> Result<Tensor> {
        let result = if train_flag {
            let mask_tensor = &Tensor::randn(0f32, 1f32, x.clone().shape(), x.clone().device())?.maximum(self.dropout_ratio)?;
            self.mask = Some(Var::from_tensor(&mask_tensor.clone())?);
            // self.mask.unwrap().set();
            x.clone() * mask_tensor.clone()
        } else {
            x.clone() * (self.dropout_ratio.neg() + 1.0)
        };
        
        result
    }

    fn backward(self, dout: Tensor) -> Result<Tensor> {
        dout * self.mask.unwrap().as_tensor()
    }
}

fn deep_learning_simple_dropout_demo(device: &Device) -> Result<()>{
    let x = Tensor::randn(0f32, 1f32, (3,4), device)?;
    let mut dropout = SimpleDropout::_init_(0.12);
    let forward_ = &dropout.forward(x.clone(), true)?;
    println!("Dropout forward={}", forward_);
    let backward_ = &dropout.backward(x.clone())?;
    println!("Dropout backward={}", backward_);

    Ok(())
}

fn deep_learning_superparam_demo(device: &Device) -> Result<()>{
    let weight_decay = (candle_nn::Init::Uniform {lo: -8.0, up:-4.0}.var((3,4), DType::F32, device)?.as_tensor() * 10.0)?;
    let lr = (candle_nn::Init::Uniform {lo:-6.0, up:-2.0}.var((3,4), DType::F32, device)?.as_tensor() * 10.0)?;

    println!("weight decay ={}, lr={}", weight_decay, lr);
    Ok(())
}

fn conv_neural_networks(device: &Device) -> Result<()> {
    let h = 32;
    let w = 48;

    let input_2d_data = Tensor::randn(0f32,1., (h,w), device)?;
    let output_2d_data = Var::from_tensor(&Tensor::zeros((h,w), DType::F32, device)?)?;
    let kernal = Tensor::randn(0f32, 1., (3,3), device)?;
    let padding = Tensor::zeros((h+2, w+2), DType::F32, device)?;
    let padding = padding.slice_assign(&[1..=h, 1..=w], &input_2d_data.clone())?;
    println!("after slice_assign: {}", padding);
    for i in (0..h) {
        for j in (0..w) {
            let window = padding.clone().i((i..i+3, j..j+3))?;
            // println!("(kernal*window)={}", (kernal.clone() * window.clone())?);
            let window_sum = (kernal.clone()*window)?.sum_all()?.reshape((1,1))?;
            // println!("(window_sum)={}", window_sum.clone());
            let output_2d_data_tensor = output_2d_data.slice_assign(&[i..i+1,j..j+1],&window_sum.clone())?;
            output_2d_data.set(&output_2d_data_tensor)?
        }
    }

    println!("output 2d data:{}", output_2d_data.clone());

    Ok(())
}

///
// def conv2D(input_2Ddata, kern):
//     (h, w) = input_2Ddata.shape # 输入数据的高度和宽度
//     (kern_h, kern_w) = kern.shape # 卷积核的高度和宽度
//     padding_h = (kern_h-1)//2
//     padding_w = (kern_w-1)//2
//     padding = np.zeros(shape = (h+2＊padding_h, w+2＊padding_w))
//     # 0填充
//     padding[padding_h:-padding_h, padding_w:-padding_w] =
//         input_2Ddata
//     output_2Ddata = np.zeros(shape = (h, w)) # 输出数据的尺寸和
//         输入数据一样
//
//     for i in range(h):
//         for j in range(w):
//             window = padding[i:i+kern_h, j:j+kern_w]
//             # 局部窗口
//             output_2Ddata[i, j] = np.sum(kern＊window) # 内积
//     return output_2Ddata
// ##########################################################
//
// h = 32 # 输入数据的高度
// w = 48 # 输入数据的宽度
// in_d = 12 # 输入数据的深度
// out_d = 24 # 输出数据的深度
// input_3Ddata = np.random.randn(h, w, in_d)
// output_3Ddata = np.zeros(shape = (h, w, out_d))
//
// (kern_h, kern_w) = (3, 3) # 或者(5, 5)
// kerns = np.random.randn(out_d, kern_h, kern_w, in_d)
// # 4D卷积核
// bias = np.random.randn(out_d) # 1D偏置
//
// for m in range(out_d): # 每一个输出2D 数据
//     for k in range(in_d): # 每一个输入2D数据
//         input_2Ddata = input_3Ddata[:, :, k] # 第k个输入2D数据
//         kern = kerns[m, :, :, k]
//         output_3Ddata[:, :, m] += conv2D(input_2Ddata, kern)
//         # 加上每个卷积结果
//     output_3Ddata[:, :, m] += bias[m] # 每个输出2D 数据只有
//         一个偏置

fn conv2D(input_2d_data: Tensor, kern: Tensor, device: &Device) -> Result<Tensor> {
    let (h, w) = (input_2d_data.clone().dims()[0], input_2d_data.clone().dims()[1]);
    let (kern_h, kern_w) = (kern.clone().dims()[0], kern.clone().dims()[1]);
    let padding_h = (kern_h - 1);
    let padding_w = (kern_w - 1);
    let padding = Tensor::zeros((h+2*padding_h, w+2*padding_w),DType::F32, device)?;
    let padding = padding.clone().slice_assign(&[padding_h..padding.clone().dims()[0]-padding_h, padding_w..padding.clone().dims()[1]-padding_w], &input_2d_data.clone())?;

    let output_2d_data = Var::from_tensor(&Tensor::zeros((h,w), DType::F32, device)?)?;

    for i in (0..h) {
        for j in (0..w) {
            let window = padding.clone().i((i..i+kern_h, j..j+kern_w))?;
            // println!("(kernal*window)={}", (kernal.clone() * window.clone())?);
            let window_sum = kern.clone().broadcast_mul(&window.reshape((window.clone().dims()[0], window.clone().dims()[1],1))?)?.sum_all()?.reshape((1,1))?;
            // println!("(window_sum)={}", window_sum.clone());
            let output_2d_data_tensor = output_2d_data.slice_assign(&[i..i+1,j..j+1],&window_sum.clone())?;
            output_2d_data.set(&output_2d_data_tensor)?
        }
    }

    Ok(output_2d_data.as_detached_tensor())

}

///
/// 3D特征图表示为 [H×W×D]，其中H是高度，W是宽度，D是深度。
/// 3D特征图可以看作D个2D数据，每个2D数据的尺寸均是 [H×W]i，称为特征图，3D特征图总共有 D个特征图
/// 
fn conv_neural_networks_v2(device: &Device) -> Result<()> {
    let h = 32;
    let w = 48;
    let in_d = 12;
    let out_d = 24;

    let input_3d_data = Tensor::randn(0f32, 1., (h,w,in_d), device)?;
    let output_3d_data = Var::from_tensor(&Tensor::randn(0f32, 1., (h,w,out_d), device)?)?;
    let (kern_h,  kern_w) = (3,3);
    let kerns = Tensor::randn(0f32, 1., (out_d, kern_h, kern_w, in_d), device)?;
    let bias = Tensor::randn(0f32, 1., (out_d), device)?;

    for m in (0..out_d) {
        for k in (0..in_d)  {
            let input_2d_data = input_3d_data.clone().i((0..input_3d_data.clone().dims()[0],0..input_3d_data.clone().dims()[1],k))?;
            let kern = kerns.clone().i((m,0..kerns.clone().dims()[1], 0..kerns.clone().dims()[2], k))?;
            let output_3d_add_2d_data =
                (output_3d_data.clone().i((0..output_3d_data.clone().dims()[0], 0..output_3d_data.clone().dims()[1], m))?
                    + conv2D(input_2d_data.clone(), kern.clone(), device)?)?;

            let output_3d_add_2d_bisa_data = output_3d_add_2d_data.clone().broadcast_add(&bias.clone().i(m)?)?;
            let output_3d_data_tensor = output_3d_data.clone().slice_assign(&[0..output_3d_data.clone().dims()[0], 0..output_3d_data.clone().dims()[1], m..m+1],&output_3d_add_2d_bisa_data.clone().reshape((output_3d_add_2d_bisa_data.clone().dims()[0], output_3d_add_2d_bisa_data.clone().dims()[1], 1))?)?;
            output_3d_data.set(&output_3d_data_tensor)?
        }
    }

    println!("output 3d data={}", output_3d_data.clone());
    Ok(())
}

fn main() {
    println!("deep learning get started...");
    let device = Device::cuda_if_available(0).unwrap();
    conv_neural_networks_v2(&device).unwrap();
    // conv_neural_networks(&device).unwrap();
    // deep_learning_superparam_demo(&device).unwrap();

    // deep_learning_simple_dropout_demo(&device).unwrap();
    // weight_decry_demo(&device).unwrap()
    // one_hot_cold_demo(&device).unwrap();
    // two_layer_deep_learning_demo_v2(&device).unwrap();
    // two_layer_deep_learning_demo(&device).unwrap();

    // backward_derivate_and_learning(&device).unwrap()

    // deep_learning_twolayer_net_demo(&device).unwrap()
    // deep_learning_v2_demo(&device).unwrap();
    // deep_learning_demo(&device).unwrap();
    // multiply_preceptron_demo(&device).unwrap();
    // naive_perceptron_demo(&device).unwrap();
}