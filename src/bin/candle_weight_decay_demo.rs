use candle_core::quantized::QMatMul::TensorF16;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, Var};
use candle_nn::{Optimizer, SGD, VarBuilder, VarMap, loss, ops, Init};
use futures::StreamExt;
use rand::prelude::SliceRandom;
use std::iter::{Enumerate, Zip};
use std::ops::Sub;
use std::slice::Iter;
use candle_nn::Init::Const;
use candle_nn::var_builder::SimpleBackend;
use clap::builder::Resettable::Reset;

///
/// """生成y=Xw+b+噪声"""
///
fn synthetic_data(
    w: &Tensor,
    b: Option<&Tensor>,
    num_examples: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let X = Tensor::randn(0f32, 1f32, (num_examples, w.dim(0)?), device)?;
    let y = match b {
        Some(b) => Tensor::matmul(&X, w)?.broadcast_add(&b.clone())?,
        None => Tensor::matmul(&X, w)?,
    };
    let y = (&y + Tensor::randn(0f32, 0.01, y.clone().shape(), device)?)?;

    Ok((X, y.reshape(((), 1))?))
}

fn load_array(
    train_data: (Tensor, Tensor),
    batch_size: usize,
    is_train: bool,
) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let (trains, labels) = train_data;
    let n_batches = trains.dim(0)? / batch_size;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    let mut trains_vec = Vec::with_capacity(n_batches);
    let mut labels_vec = Vec::with_capacity(n_batches);
    if is_train {
        batch_idxs.shuffle(&mut rand::thread_rng());
    }
    for batch_idx in batch_idxs.iter() {
        let batch_train = trains.narrow(0, batch_size * batch_idx, batch_size)?;
        trains_vec.push(batch_train.to_owned());
        let batch_train_labels = labels.narrow(0, batch_size * batch_idx, batch_size)?;
        labels_vec.push(batch_train_labels.to_owned());
    }

    Ok((trains_vec.to_owned(), labels_vec.to_owned()))
}

fn init_params(num_inputs: usize, device: &Device) -> Result<(Var, Option<Var>)> {
    let w = Tensor::randn(0f32, 1f32, (num_inputs, 1), device)?;
    let w = Var::from_tensor(&w)?;

    let b = Tensor::ones(1, DType::F32, device)?;
    let b = Var::from_tensor(&b)?;

    Ok((w, Some(b)))
}

///
/// L2范数惩罚
///
fn l2_penalty(w: Var) -> Result<Tensor> {
    let w_pow = w.as_tensor().powf(2f64)?;
    let w_sum = w_pow.sum_all()?;
    let w_div = (w_sum / 2f64)?;

    Ok(w_div)
}

///
/// """均方损失"""
///
fn squared_loss(y_hat: Tensor, y: Tensor) -> Result<Tensor> {
    let result = (((y_hat.clone() - y.reshape(y_hat.clone().shape()))? * 2f64)? / 2f64);
    // println!("result squared_loss: {:?}", result.as_ref().unwrap().shape());
    result
}

/// 线性模型
fn linereg(X: Tensor, w:Var, b: Option<Var>) -> Result<Tensor> {
    if b.is_some() {
        Tensor::matmul(&X, &w)?.broadcast_add(&b.unwrap())
    } else {
        Tensor::matmul(&X, &w)
    }
}

/// 优化函数
fn sgd(params: Vec<Var>, grad: Tensor, lr: f64, batch_size: usize) -> Result<()> {
    for mut param in  params {
        // println!("param.as_tensor()-shape:{:?}; grad shape:{:?}", param.as_tensor().shape(), grad.shape());
        if grad.dims().len() == param.dims().len() {
            param.set(&(param.as_tensor().t()?.broadcast_sub(&((lr*&grad)? / (batch_size as f64))?)?));
        } else {
            param.set(&(param.as_tensor().repeat(param.shape())?.broadcast_sub(&((lr*&grad)? / (batch_size as f64))?)?));
        }
        // param -= lr * &grad? / batch_size
    }
    Ok(())
}

fn evaluate_loss(
    lamdb: f64,
    w: Var,
    b: Var,
    data_iter: Enumerate<Zip<Iter<Tensor>, Iter<Tensor>>>,
    net: impl Fn(Tensor, Var, Option<Var>) -> Result<Tensor>,
    activation: candle_nn::Activation,
    optimizer: &mut SGD,
    loss: impl Fn(Tensor, Tensor) -> Result<Tensor>,
    batches: usize,
    lr: f64,
) -> Result<()> {
    for (data, label) in data_iter.map(|(i, (data, label))|{ (data, label)  }).into_iter() {
        let logits = net(data.detach(), w.clone(), Some(b.clone()))?;
        // println!("logits shape = {:?}", logits.shape());
        let y = label.detach();
        // println!("label shape={:?}; y shape = {:?}",label.shape(), y.shape());
        let loss = loss(logits, y)?;

        println!("Avg loss of {:.3}", (loss.sum_all()?.to_vec0::<f32>()? / loss.elem_count() as f32));
        // let relu = activation.forward(&logits).unwrap();
        // let loss = (loss(logits, label.detach()).unwrap().broadcast_add(
        //     &(l2_penalty(w.clone()).unwrap() * lamdb).unwrap()
        // )).unwrap();
        // let grad_store = loss.backward().unwrap();
        // let grad = grad_store.get(data).expect("no grad");
        // sgd(vec![w.to_owned(),b.to_owned()], grad.detach(), lr, batches).unwrap();
    }

    Ok(())
}

fn train(
    lamda: f64,
    train_iter: Enumerate<Zip<Iter<Tensor>, Iter<Tensor>>>,
    batches: usize,
    num_inputs: usize,
    device: &Device,
) -> Result<()> {
    let (w_var, b_var) = init_params(num_inputs, device)?;

    let w = w_var.as_detached_tensor().t()?;
    let b = match b_var.clone() {
        Some(b) => Some(b.as_detached_tensor()),
        None => None,
    };

    // net, loss
    //let (net, loss) = (candle_nn::Linear::new(w.clone(), b.clone()), squared_loss);
    let (net, loss) = (linereg, squared_loss);
    let (num_epochs, lr) = (100, 0.003);
    let mut varmap = VarMap::new();
    // varmap.set([("w", &w.clone())].into_iter())?;
    // varmap.set([("b", &b.clone().unwrap())].into_iter())?;
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let mut sgd = SGD::new(varmap.all_vars(), lr)?;
    
    let activation = candle_nn::Activation::Relu;
    for epoch in 0..num_epochs {
        // println!("epoch {}/{}", epoch + 1, num_epochs);
        train_iter.to_owned().for_each(|(i, (train, label))|{
            let l = loss(net(train.clone(), w_var.clone(), b_var.clone()).unwrap(), label.detach()).unwrap();
            let l = (l.broadcast_add(&(l2_penalty(w_var.clone()).unwrap() * lamda).unwrap())).unwrap();
            l.sum_all().unwrap().backward().unwrap();

            if(epoch + 1) % 5 == 0 {
                evaluate_loss(0f64, w_var.clone(), b_var.clone().unwrap(),train_iter.clone(), net, activation, &mut sgd, loss, batches, lr);
            }
        })
    }
    
    println!("w的L2范数是：{:?}", w.sqr()?.sum([0,1])?.sqrt()?);

    Ok(())
}

fn train_concise(
    lamda: f64,
    train_iter: Enumerate<Zip<Iter<Tensor>, Iter<Tensor>>>,
    batches: usize,
    num_inputs: usize,
    device: &Device,
) -> Result<()>{
    let device_inner = device.clone();

    let (w_var, b_var) = init_params(num_inputs, device)?;
    let w_inner = w_var.clone();
    let w_outer = w_var.clone();
    let inner_fn = move |tensor: &Tensor| -> Result<Tensor> {
        // 重置
        let dims = tensor.dims();
        w_inner.clone().set(&Tensor::randn(0f32, 1f32, (dims[1], dims[0]), &device_inner)?);
        Ok(tensor.detach())
    };
    let (num_epochs, lr) = (100, 0.003);
    let mut varmap = VarMap::new();
    varmap.get(1,"weight_decay", Const(lamda), DType::F64, device);
    // varmap.set([("w", &w.clone())].into_iter())?;
    // varmap.set([("b", &b.clone().unwrap())].into_iter())?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let net = candle_nn::sequential::seq()
        .add(candle_nn::linear::linear(num_inputs, 1, vb)?)
        .add_fn(inner_fn)
        ;
    let mut sgd = SGD::new(varmap.all_vars(), lr)?;
    let activation = candle_nn::Activation::Relu;
    let loss = candle_nn::loss::mse;

    for epoch in 0..num_epochs {
        // println!("epoch {}/{}", epoch + 1, num_epochs);
        train_iter.to_owned().for_each(|(i, (train, label))|{
            let logits = net.forward(&train.clone()).unwrap();
            let l = loss(&logits, label).unwrap();
            // let l = (l * lamda).unwrap();
            let mean_all = l.mean_all().unwrap().backward().unwrap();
            sgd.step(&mean_all);

            if(epoch + 1) % 5 == 0 {
                for (data, label) in train_iter.clone().map(|(i, (data, label))|{ (data, label)  }).into_iter() {
                    let logits = net.forward(data).unwrap();
                    // println!("logits shape = {:?}", logits.shape());
                    let y = label;
                    // println!("label shape={:?}; y shape = {:?}",label.shape(), y.shape());
                    let loss = loss(&logits, y).unwrap();

                    println!("Avg loss of {:.3}", (loss.sum_all().unwrap().to_vec0::<f32>().unwrap() / loss.elem_count() as f32));
                }
            }
        })
    }

    println!("w的L2范数是：{:?}", w_outer.sqr()?.sum([0,1])?.sqrt()?);

    
    Ok(())
}

fn train_concise_v1(
    lamda: f64,
    train_iter: Enumerate<Zip<Iter<Tensor>, Iter<Tensor>>>,
    num_inputs: usize,
    device: &Device,
) -> Result<()>{
    // 定义网络结构
    struct Net {
        linear: candle_nn::Linear,
    }

    impl Net {
        fn new(num_inputs: usize, device: &Device) -> Result<Self> {
            // 使用正态分布初始化权重，零初始化偏置
            let (w_var, b_var) = init_params(num_inputs, device)?;
            let ws = w_var.detach();
            let bs = b_var.unwrap().detach();
            let linear = candle_nn::Linear::new(ws.t()?, Some(bs));
            Ok(Self { linear })
        }
    }

    impl Module for Net {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            self.linear.forward(x)
        }
    }
    
    let mut varmap = VarMap::new();
    varmap.get((num_inputs, 1),"weight", Init::Randn {mean:0., stdev:1.}, DType::F32, device);
    varmap.get(1,"bias", Const(0.), DType::F32, device);
    // 初始化模型
    let mut net = Net::new(num_inputs, &device)?;
    // 创建优化器（所有参数）
    let params = vec![
        Var::from_tensor(&net.linear.weight().clone())?,
        Var::from_tensor(&net.linear.bias().unwrap().clone())?,
    ];

    let lr = 0.003;

    let paramsAdamw = candle_nn::ParamsAdamW {
        lr: 0.1,
        weight_decay: lamda,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), paramsAdamw)?;
    let num_epochs = 100;

    for epoch in 0..num_epochs {
        // println!("epoch {}/{}", epoch + 1, num_epochs);
        train_iter.to_owned().for_each(|(i, (train, label))|{
            let logits = net.forward(&train.clone()).unwrap();
            let mse = candle_nn::loss::mse(&logits, label).unwrap();
            let l = (mse.broadcast_add(&(l2_penalty(Var::from_tensor(net.linear.weight()).unwrap()).unwrap() * lamda).unwrap())).unwrap();
            let loss = mse;
            let mean_all = loss.mean_all().unwrap().backward().unwrap();
            optimizer.step(&mean_all);

            if(epoch + 1) % 5 == 0 {
                for (data, label) in train_iter.clone().map(|(i, (data, label))|{ (data, label)  }).into_iter() {
                    let logits = net.forward(data).unwrap();
                    let y = label;
                    let mse = candle_nn::loss::mse(&logits, y).unwrap();
                    let loss = mse;

                    println!("Avg loss of {:.3}", (loss.sum_all().unwrap().to_vec0::<f32>().unwrap() / loss.elem_count() as f32));
                }
            }
        })
    }

    println!("w的L2范数是：{:?}", net.linear.weight().sqr()?.sum([0,1])?.sqrt()?);

    Ok(())
}

/// 生成测试数据，要求如下：
/// - 标签均值=0 标准差=0.01的高斯噪声破坏
/// - 数据维度=200 样本集=20 突出过拟合
fn generate_dataset(device: &Device) -> Result<()> {
    struct DataSet {
        train_data: Tensor,
        train_label: Tensor,
        test_data: Tensor,
        test_label: Tensor,
    }

    let (n_train, n_test, num_inputs, batch_size) = (20, 100, 200, 5);
    let true_w = Tensor::ones((num_inputs, 1), DType::F32, device)? * 0.01f64;
    let true_w = true_w?.to_dtype(DType::F32)?;
    let true_b = Tensor::new(&[0.05f32], device)?;
    let train_data = synthetic_data(&true_w, Some(&true_b), n_train, device)?;

    // X_shape=(20, 200) Y_shape=(20,1)
    println!("X={}, Y={}", train_data.0, train_data.1);
    let train_iter = load_array(train_data, batch_size, true)?;
    // train_shape=(20, 200), label_shape=(20,1)
    // println!("train={}, label={}", train_iter.0, train_iter.1);
    // train_iter.0.iter().zip(train_iter.1.iter()).enumerate().for_each(|(i, data)| {
    //     println!("Train Datas: idx={:?}, data={}, label={}", i, data.0, data.1);
    // });
    // train(3f64,
    //     train_iter.0.iter().zip(train_iter.1.iter()).enumerate(),
    //     batch_size,
    //     num_inputs,
    //     device,
    // )?;

    train_concise_v1(
        3f64,
        train_iter.0.iter().zip(train_iter.1.iter()).enumerate(),
        num_inputs,
        device
    )?;
    
    // train_concise(0f64,
    //       train_iter.0.iter().zip(train_iter.1.iter()).enumerate(),
    //       batch_size,
    //       num_inputs,
    //       device,
    // )?;

    // 测试集
    // let test_data = synthetic_data(&true_w, Some(&true_b), n_test, device)?;
    // println!("test_data={}, {}", test_data.0, test_data.1);
    // let test_iter = load_array(test_data, batch_size, false)?;
    
    
    // println!("test_iter={}, {}", test_iter.0, test_iter.1);
    // test_iter
    //     .0
    //     .iter()
    //     .zip(test_iter.1.iter())
    //     .enumerate()
    //     .for_each(|(i, data)| {
    //         println!(
    //             "Train datas: idx={:?}, data={}, label={}",
    //             i, data.0, data.1
    //         );
    //     });

    // let params = init_params(num_inputs, device)?;

    Ok(())
}

fn main() {
    let device = Device::Cpu;
    generate_dataset(&device).unwrap();
}
