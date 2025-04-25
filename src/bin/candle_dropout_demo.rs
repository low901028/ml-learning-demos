use std::any::Any;
use std::ops::Deref;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_datasets::vision::Dataset;
use candle_nn::{Init, Optimizer, Sequential, VarBuilder, VarMap};
use candle_nn::Init::Const;
use rand::prelude::SliceRandom;

fn dropout_laypout(X: Tensor, dropout: f32, device: &Device) -> Result<Tensor> {
    assert!(dropout >= 0.0 && dropout <= 1.0);

    if f32::eq(&dropout, &1.0) {
        return Tensor::zeros_like(&X);
    }

    if f32::eq(&dropout, &0.0) {
        return Ok((X));
    }

    let gather_tensor = Tensor::new(&[dropout], device)?;
    let mask = Tensor::rand(0.0f32, 1.0f32, X.shape(), device)?
        .broadcast_gt(&gather_tensor)?
        .to_dtype(DType::F32)?;
    // println!("=== mask shape={}", mask);

    let result = ((mask * X)? / ((1.0 - dropout) as f64))?;
    // println!("=== result shape={:?}", result.clone().shape());

    Ok(result.clone())
}

fn load_fashion_mnist() -> Result<Dataset> {
    let home_dir = std::path::Path::new("datas");
    let dataset = candle_datasets::vision::mnist::load_dir(home_dir)?;
    println!("train images shape={:?}; labels={:?}", dataset.train_images.shape(), dataset.train_labels.shape());
    println!("test images shape={:?}; labels={:?}", dataset.test_images.shape(), dataset.test_labels.shape());
    Ok(dataset)
}

fn fashion_mnist_process(batch_size: usize, is_train: bool , device: Device) -> Result<Vec<(Tensor, Tensor)>>{
    let dataset = load_fashion_mnist()?;
    let mut datas: Vec<(Tensor, Tensor)> = vec![];
    if is_train {
        // train数据集
        let train_images = dataset.train_images;
        let train_labels = dataset.train_labels.to_device(&device)?;
        let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&device)?;

        let n_batches = train_images.dim(0)? / batch_size;
        let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
        /// 基于candle-core中的narrow来进行随机小批量测试
        batch_idxs.shuffle(&mut rand::thread_rng());
        for batch_idx in batch_idxs.iter() {
            let batch_train_images = train_images.narrow(0, batch_size*batch_idx, batch_size)?;
            let batch_train_labels = train_labels.narrow(0, batch_size*batch_idx, batch_size)?;
            //
            datas.push((batch_train_images, batch_train_labels));
        }
    } else {
        // test数据集
        let test_images = dataset.test_images;
        let test_labels = dataset.test_labels;
        let test_labels = test_labels.to_dtype(DType::U32)?.to_device(&device)?;

        datas.push((test_images, test_labels));
        // let n_batches = test_images.dim(0)? / batch_size;
        // let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
        // /// 基于candle-core中的narrow来进行随机小批量测试
        // batch_idxs.shuffle(&mut rand::thread_rng());
        // for batch_idx in batch_idxs.iter() {
        //     let batch_train_images = test_images.narrow(0, batch_size*batch_idx, batch_size)?;
        //     let batch_train_labels = test_labels.narrow(0, batch_size*batch_idx, batch_size)?;
        //     //
        //     datas.push((batch_train_images, batch_train_labels));
        // }
    };


    Ok(datas)
}

fn train(device: &Device) -> Result<()> {
    // let X = Tensor::arange(0f32, 16., &device.clone())?.reshape((2, 8))?;
    // println!("Tensor X={}", X);
    // println!(
    //     "dropout_laypout(X, 0.0): {}",
    //     dropout_laypout(X.clone(), 0.0, device)?
    // );
    // println!(
    //     "dropout_laypout(X, 0.5): {}",
    //     dropout_laypout(X.clone(), 0.5, device)?
    // );
    // println!(
    //     "dropout_laypout(X, 1.): {}",
    //     dropout_laypout(X.clone(), 1.0, device)?
    // );

    let (num_inputs, num_outputs, num_hiddens1, num_hiddens2) = (784, 10, 256, 256);
    let (dropout1, dropout2) = (0.2, 0.5);

    struct Net {
        num_inputs: usize,
        is_training: bool,
        linner1: candle_nn::Linear,
        linner2: candle_nn::Linear,
        linner3: candle_nn::Linear,
        relu: candle_nn::activation::Activation,
        dropout1: f32,
        dropout2: f32,
        device: Device,
    }

    impl Net {
        fn new(
            num_inputs: usize,
            num_outputs: usize,
            num_hiddens1: usize,
            num_hiddens2: usize,
            vb: VarBuilder,
            is_training: bool,
            dropout1: f32,
            dropout2: f32,
            device: Device,
        ) -> Result<Self> {
           let linear1_w = vb.get_with_hints((num_hiddens1,num_inputs), "linear1_weight", Init::Randn {mean:0., stdev: 1.})?;
           let linear1_b = vb.get_with_hints((1,1), "linear1_bias", Const(0.))?;

            let linear2_w = vb.get_with_hints((num_hiddens2,num_hiddens1), "linear2_weight", Init::Randn {mean:0., stdev: 1.})?;
            let linear2_b = vb.get_with_hints((1,1), "linear2_bias", Const(0.))?.t()?;

            let linear3_w = vb.get_with_hints((num_outputs,num_hiddens2), "linear3_weight", Init::Randn {mean:0., stdev: 1.})?;
            let linear3_b = vb.get_with_hints((1,1), "linear3_bias", Const(0.))?;
            Ok(Self {
                num_inputs: num_inputs,
                is_training:  is_training,
                // linner1: candle_nn::linear(num_inputs, num_hiddens1, vb.clone())?,
                // linner2: candle_nn::linear(num_hiddens1, num_hiddens2, vb.clone())?,
                // linner3: candle_nn::linear(num_hiddens2, num_outputs, vb.clone())?,
                linner1: candle_nn::Linear::new(linear1_w, Some(linear1_b)),
                linner2: candle_nn::Linear::new(linear2_w, Some(linear2_b)),
                linner3: candle_nn::Linear::new(linear3_w, Some(linear3_b)),
                relu: candle_nn::Activation::Relu,
                dropout1: dropout1,
                dropout2: dropout2,
                device: device,
            })
        }
    }

    impl candle_nn::Module for Net {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let H1 =
                self.relu.forward(
                    self.linner1
                    .forward(
                        xs
                        //xs.reshape((self.num_inputs,()))?.as_ref()
                    )?
                    .as_ref(),
            )?;

            let H1 = if self.is_training == true {
                dropout_laypout(H1, self.dropout1, &self.device)?
            } else {
                H1
            };

            let H2 = self.relu.forward(self.linner2.forward(&H1)?.as_ref())?;
            let H2 = if self.is_training {
                dropout_laypout(H2, self.dropout2, &self.device)?
            } else {
                H2
            };

            let out = self.linner3.forward(&H2)?;
            Ok(out)
        }
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device.clone());

    let net = Net::new(
        num_inputs,
        num_outputs,
        num_hiddens1,
        num_hiddens2,
        vb,
        true,
        dropout1,
        dropout2,
        device.clone()
    )?;

    let (num_epochs, lr, batch_size) = (10, 0.5, 256);
    let loss = candle_nn::loss::cross_entropy;
    let mut trainer= candle_nn::optim::SGD::new(varmap.all_vars(), lr)?;

    let train_data_iter = fashion_mnist_process(batch_size, true, device.clone())?;
    let test_data_iter = fashion_mnist_process(batch_size, false, device.clone())?;
    let mut train_metrics = (0.0,1);
    for epoch in 0..num_epochs {
        for (i,(X, y)) in train_data_iter.iter().enumerate() {
            let y_hat = net.forward(X)?;
            let l = loss(&y_hat, &y)?;
            trainer.backward_step(&l)?;

            let sum_all = l.sum_all()?.to_scalar::<f32>()? as f64;
            train_metrics.0 += sum_all;
            train_metrics.1 += i;
            let avg_loss = sum_all / (batch_size as f64);
            // println!("[inner] batchs={:?}, avg loss={:?}", i, avg_loss);
        }

        let avg_loss = train_metrics.0 / (batch_size * train_metrics.1) as f64;

        for (i,(X, y)) in test_data_iter.iter().enumerate() {
            let test_ys = net.forward(&X)?;
            let sum_ok = test_ys
                .argmax(D::Minus1)?
                .eq(y)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / y.dims1()? as f32;
            println!("avg_losss={:?}, test accuracy: {:?}", avg_loss, test_accuracy);
        }
    }
    for var in varmap.clone().all_vars() {
        println!("{}", var);
    }
    
    Ok(())
}

fn sequential_train(device: &Device) -> Result<()> {
    let (num_inputs, num_outputs, num_hiddens1, num_hiddens2) = (784, 10, 256, 256);
    let (dropout1, dropout2) = (0.2, 0.5);
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device.clone());
    
    let linear1_w = vb.get_with_hints((num_hiddens1,num_inputs), "linear1_weight", Init::Randn {mean:0., stdev: 1.})?;
    let linear1_b = vb.get_with_hints((1,1), "linear1_bias", Const(0.))?;

    let linear2_w = vb.get_with_hints((num_hiddens2,num_hiddens1), "linear2_weight", Init::Randn {mean:0., stdev: 1.})?;
    let linear2_b = vb.get_with_hints((1,1), "linear2_bias", Const(0.))?.t()?;

    let linear3_w = vb.get_with_hints((num_outputs,num_hiddens2), "linear3_weight", Init::Randn {mean:0., stdev: 1.})?;
    let linear3_b = vb.get_with_hints((1,1), "linear3_bias", Const(0.))?;
    let mut is_train = true;
    
    
    let net = |is_train: bool| -> Result<Sequential> {
        let net = candle_nn::sequential::seq()
            .add(candle_nn::Linear::new(linear1_w.clone(), Some(linear1_b.clone())))
            .add(candle_nn::activation::Activation::Relu)
            .add_fn(move |tensor| {
                let dropout = candle_nn::Dropout::new(dropout1);
                dropout.forward(tensor, is_train)
            })
            .add(candle_nn::Linear::new(linear2_w.clone(), Some(linear2_b.clone())))
            .add(candle_nn::activation::Activation::Relu)
            .add_fn(move |tensor| {
                let dropout = candle_nn::Dropout::new(dropout2);
                dropout.forward(tensor, is_train)
            })
            .add(candle_nn::Linear::new(linear3_w.clone(),  Some(linear3_b.clone())))
            ;
        
        Ok(net)
    };
    
    

    let (num_epochs, lr, batch_size) = (10, 0.5, 256);
    let loss = candle_nn::loss::cross_entropy;
    let mut trainer= candle_nn::optim::SGD::new(varmap.all_vars(), lr)?;

    let train_data_iter = fashion_mnist_process(batch_size, true, device.clone())?;
    let test_data_iter = fashion_mnist_process(batch_size, false, device.clone())?;
    let mut train_metrics = (0.0,1);
    for epoch in 0..num_epochs {
        is_train = true;
        for (i,(X, y)) in train_data_iter.iter().enumerate() {
            let y_hat = net(is_train)?.forward(X)?;
            let l = loss(&y_hat, &y)?;
            trainer.backward_step(&l)?;

            let sum_all = l.sum_all()?.to_scalar::<f32>()? as f64;
            train_metrics.0 += sum_all;
            train_metrics.1 += i;
            let avg_loss = sum_all / (batch_size as f64);
            // println!("[inner] batchs={:?}, avg loss={:?}", i, avg_loss);
        }

        let avg_loss = train_metrics.0 / (batch_size * train_metrics.1) as f64;
        is_train = false;
        for (i,(X, y)) in test_data_iter.iter().enumerate() {
            let test_ys = net(is_train)?.forward(&X)?;
            let sum_ok = test_ys
                .argmax(D::Minus1)?
                .eq(y)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / y.dims1()? as f32;
            println!("avg_losss={:?}, test accuracy: {:?}", avg_loss, test_accuracy);
        }
    }
    for var in varmap.clone().all_vars() {
        println!("{}", var);
    }
    
    
    Ok(())
}

fn main() {
    let device = Device::Cpu;
    sequential_train(&device).unwrap();
    // train(&device).unwrap();
}
