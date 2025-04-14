use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Error, IndexOp, Module, ModuleT, NdArray, Result, Tensor, Var, D};
use candle_nn::Init::Const;
use candle_nn::ops::softmax;
use candle_nn::{AdamW, Linear, Optimizer, ParamsAdamW, SGD, VarBuilder, VarMap, linear, Sequential};
use parquet::file::reader::{FileReader, SerializedFileReader};
use plotters::prelude::full_palette::PURPLE;
use plotters::prelude::{BLACK, BLUE, BitMapBackend, ChartBuilder, Circle, Color, DiscreteRanged, EmptyElement, GREEN, IntoDrawingArea, IntoFont, IntoLinspace, IntoSegmentedCoord, LabelAreaPosition, LineSeries, PathElement, PointSeries, RED, Text, WHITE, BitMapElement};
use rand::prelude::SliceRandom;
use std::fs::File;
use std::io::{self,BufReader, ErrorKind, Read, Write};
use std::ops::{Deref, Index, Mul};
use std::rc::Rc;
use std::sync::Arc;
use flate2::read::GzDecoder;
use image::{imageops::FilterType, DynamicImage, GrayImage, ImageBuffer, ImageFormat, Rgb, Rgba};
use byteorder::{BigEndian, ReadBytesExt};
use candle_core::op::Op;
use candle_datasets::vision::cifar::load_dir;
use candle_datasets::vision::Dataset;
use candle_nn::init::Init;
use candle_transformers::models::mimi::candle;
use futures::{StreamExt, TryStreamExt};
use futures::executor::block_on;
use plotters::backend::{PixelFormat, RGBPixel};

/// 结合plotters绘制正态分布图
///
fn candle_basic_model_demo(device: &Device) -> Result<()> {
    let n = 1000;
    let a = Tensor::arange(0f32, n as f32, device)?;
    let b = Tensor::arange(0f32, n as f32, device)?;

    let c = (a + b)?;
    // println!("c = {}", c);

    // 正态分布
    fn noraml(x: f32, mu: f32, sigma: f32) -> f32 {
        let p = 1.0 / f32::sqrt(2.0 * std::f32::consts::PI * sigma.powf(2.0));
        p * f32::exp(-0.5 / sigma.powf(2.0) * (x - mu).powf(2.0))
    }

    let x = 10.0f32;
    let mu = 0.0f32;
    let sigma = 1.0f32;
    println!("p(x={}) = {}", x, noraml(x, mu, sigma));

    let OUT_FILE_NAME = "basic_model_demos.png";
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (1376, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let root_area = root_area.titled("线性回归", ("sans-serif", 20)).unwrap();

    let x_axis = (-7.0f32..7.0).step(0.01);
    let params = [(0, 1), (0, 2), (3, 1)];

    let mut chart = ChartBuilder::on(&root_area)
        .margin(5)
        .caption("正态分布图", ("sans-serif", 20))
        .set_label_area_size(LabelAreaPosition::Top, 60)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        // .x_label_area_size(30)
        // .y_label_area_size(30)
        // .top_x_label_area_size(30)
        // .right_y_label_area_size(30)
        .build_cartesian_2d(-7.0f32..7.0f32, 0.0f32..0.4)
        .unwrap();

    chart
        .configure_mesh()
        // .disable_x_mesh()
        // .disable_y_mesh()
        .x_labels(20)
        .y_labels(20)
        .y_label_formatter(&|y| format!("{:.1}f", *y))
        .y_desc("P(x)")
        .x_desc("x")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            x_axis.values().map(|x| {
                (
                    x,
                    noraml(
                        x,
                        params.get(0).unwrap().0 as f32,
                        params.get(0).unwrap().1 as f32,
                    ),
                )
            }),
            &BLUE,
        ))
        .unwrap()
        .label("mean 0， std 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(
            x_axis.values().map(|x| {
                (
                    x,
                    noraml(
                        x,
                        params.get(1).unwrap().0 as f32,
                        params.get(1).unwrap().1 as f32,
                    ),
                )
            }),
            &PURPLE,
        ))
        .unwrap()
        .label("mean 0, std 2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PURPLE));

    chart
        .draw_series(LineSeries::new(
            x_axis.values().map(|x| {
                (
                    x,
                    noraml(
                        x,
                        params.get(2).unwrap().0 as f32,
                        params.get(2).unwrap().1 as f32,
                    ),
                )
            }),
            &GREEN,
        ))
        .unwrap()
        .label("mean 3, std 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    // 配置边框
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    // 保存图片
    root_area.present().unwrap();

    Ok(())
}

/// 如下的线性回归模型有三种示例
/// 简单线性模型
fn candle_basic_modle_linermodel_demo_v1(device: &Device) -> Result<()> {
    struct Model {
        f: Tensor,
        s: Tensor,
    }

    impl Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            let x = input.matmul(&self.f)?;
            // ReLU激活函数：f(x)=max(0, x)
            let x = x.relu()?;

            x.matmul(&self.s)
        }
    }

    let f = Tensor::randn(0f32, 1.0f32, (784, 100), device)?;
    let s = Tensor::randn(0f32, 1.0f32, (100, 10), device)?;
    let model = Model { f, s };

    let input = Tensor::randn(0f32, 1.0f32, (1, 784), device)?;
    let digits = model.forward(&input)?;
    println!("basic digit: {digits} digit");

    Ok(())
}

/// 加权线性回归模型
fn candle_basic_modle_linermodel_demo_v2(device: &Device) -> Result<()> {
    struct Linear {
        w: Tensor,
        b: Tensor,
    }

    impl Linear {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            let x = input.matmul(&self.w)?;
            x.broadcast_add(&self.b)
        }
    }

    // 整个模型有两个线性层
    struct Model {
        f: Linear,
        s: Linear,
    }

    impl Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            let x = self.f.forward(input)?;
            // ReLU激活函数
            let x = x.relu()?;

            self.s.forward(&x)
        }
    }
    // 线性层L1
    let w = Tensor::randn(0f32, 1.0f32, (784, 100), device)?;
    let b = Tensor::randn(0f32, 1.0f32, (100,), device)?;
    let f = Linear { w, b };
    // 线性层L2
    let w = Tensor::randn(0f32, 1.0f32, (100, 10), device)?;
    let b = Tensor::randn(0f32, 1.0f32, (10,), device)?;
    let s = Linear { w, b };
    // 模型
    let model = Model { f, s };

    let input = Tensor::randn(0f32, 1.0f32, (1, 784), device)?;
    let digits = model.forward(&input)?;
    println!(" Weighted linearit digit: {digits} digit");

    Ok(())
}

/// candle_nn中提供的LM
/// 首先要添加candle-nn依赖: cargo add --git https://github.com/huggingface/candle.git candle-nn
fn candle_basic_modle_linermodel_demo_v3(device: &Device) -> Result<()> {
    struct Model {
        f: candle_nn::Linear,
        s: candle_nn::Linear,
    }

    impl Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            // L1层
            let x = self.f.forward(input)?;
            // ReLU激活函数
            let x = x.relu()?;
            // L2层
            self.s.forward(&x)
        }
    }

    let w = Tensor::randn(0f32, 1.0f32, (100, 784), device)?;
    let b = Tensor::randn(0f32, 1.0f32, (100,), device)?;
    let f = candle_nn::Linear::new(w, Some(b));

    let w = Tensor::randn(0f32, 1.0f32, (10, 100), device)?;
    let b = Tensor::randn(0f32, 1.0f32, (10,), device)?;
    let s = candle_nn::Linear::new(w, Some(b));

    let model = Model { f, s };
    let input = Tensor::randn(0f32, 1.0f32, (1, 784), device)?;
    let digits = model.forward(&input)?;
    println!("candle_nn digit: {digits} digit");

    Ok(())
}

/// 根据构建的数据集绘制散点图
fn dataset_noise_draw_points_demo(features: &Tensor, labels: &Tensor) -> Result<()> {
    const OUT_FILE_NAME: &str = "basic-normal-dist.png";
    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let root_area = root.titled("线性模型", ("sans-serif", 20)).unwrap();

    let mut chart = ChartBuilder::on(&root_area)
        .margin(5)
        .caption("散点图", ("sans-serif", 20))
        .set_label_area_size(LabelAreaPosition::Top, 60)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .build_cartesian_2d(-4.0f32..4.0f32, -10.0f32..20.0)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .y_label_formatter(&|y| format!("{:.1}f", *y))
        .draw()
        .unwrap();

    chart
        .draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
            // 特征和标签
            // 基于zip合并，构建元组(feature, label)
            (&features.i((.., 1))?)
                .to_vec1::<f32>()?
                .into_iter()
                .zip((&labels.i((.., 0))?).to_vec1::<f32>()?.into_iter())
                // .enumerate()
                // .map(|(i, (feature, label))| {
                //     // println!("index={i}, feature={feature}, label={label}");
                //     (feature, label)
                // })
                .map(|(feature, label)| {
                    // println!("feature={feature}, label={label}");
                    (feature, label)
                })
                .collect::<Vec<_>>(), // 先获取特征数据，再根据索引获取标签
                                      // (&features.i((..,1))?).to_vec1::<f32>()?.into_iter()
                                      //     .enumerate()
                                      //     .map(|(i, feature)| {
                                      //        let binding = (&labels.i((..,0)).unwrap()).to_vec1::<f32>().unwrap();
                                      //        let label= binding.get(i).unwrap().to_owned();
                                      //        // println!("feature={feature}, label={label}");
                                      //        (feature, label)
                                      //     })
            2,
            GREEN.filled(),
        ))
        .unwrap();
    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}
/// 根据带有噪声的线性来构建数据集
/// y = Xw + b + e
fn candle_baisc_modle_dataset_noise_demo(device: &Device) -> Result<()> {
    // 线性模型
    let true_w = Tensor::new(&[2f32, 3.4], device)?;
    let true_b = 4.2f32;

    /// step-1: 生成数据集
    fn synthetic_data(
        w: &Tensor,
        b: f32,
        num_examples: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let X = Tensor::randn(0f32, 1.0f32, (num_examples, w.dim(0)?), &device)?;
        let y = (X.matmul(&w)? + b as f64)?;
        let y_clone = y.clone();
        let y = (y + Tensor::randn(0f32, 1.0f32, y_clone.shape(), &device)?)?;

        Ok((X, y.reshape(((), 1))?))
    }

    let (features, labels) = synthetic_data(&true_w.reshape(((), 1))?, true_b, 1000, device)?;
    println!("features={}, lables={}", features.i(0)?, labels.i(0)?);
    // 绘制散点图
    // dataset_noise_draw_points_demo(&features, &labels);
    /// step-2 将生成的数据集打乱，并以小批量的形式返回

    let batch_size = 10;
    // data_iter(batch_size, &features, &labels, device);
    /// step-3 初始化模型参数 由于需要grad，则需要Var来定义
    let w = Var::randn(0f32, 0.01f32, (2, 1), &device)?;
    let b = Var::zeros(1, DType::F32, &device)?;
    /// step-4 定义模型
    fn linreg(X: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        // X * w + b
        let y = (X.matmul(w)?.broadcast_add(b))?;
        Ok(y)
    }

    /// step-5 定义损失函数
    fn squared_loss(y_hat: &Tensor, y: &Tensor) -> Result<Tensor> {
        // (y_hat - y)^2/2
        let l = (y_hat - y.reshape(y_hat.shape()))?.sqr()? / 2f64;
        l
    }

    /// step-6 定义优化算法
    fn sgd(gradstore: GradStore, params: &[&Var], lr: f32, batch_size: usize) -> Result<()> {
        for param in params {
            let new_param = (param.as_tensor()
                - ((lr as f64)
                    * gradstore
                        .get(param.as_tensor())
                        .expect(format!("no grad for {}", param).as_ref()))?
                    / batch_size as f64)?;
            param.set(&new_param);
            // param.set(((param.as_tensor() - ((lr as f64) * param.as_tensor())? / batch_size as f64)?).as_ref());
            // w -= lr * w.grad() / batch_size
        }
        Ok(())
    }

    /// step-7 训练模型
    let lr = 0.03;
    let num_epochs = 3;
    let net = linreg;
    let loss = squared_loss;
    for epoch in 0..num_epochs {
        for (batch_i, (X, y)) in data_iter(batch_size, &features, &labels, &device,Some(10))?
            .into_iter()
            .enumerate()
        {
            // 前向传播
            let y_hat = net(&X, &w, &b)?;
            // 计算损失
            let l = loss(&y_hat, &y)?;
            // 反向传播
            let gradstore = l.backward()?;
            // 优化模型
            sgd(gradstore, &[&w, &b], lr, batch_size)?;
        }
        // 测试
        let train_1 = loss(&net(&features, &w, &b)?, &labels)?;
        println!("epoch {:?}, loss {:.2}", epoch + 1, train_1.mean(0)?);
        // 通过训练得到相对较优的参数w和b
        println!("w={}, b={}", w.as_tensor(), b.as_tensor());
        println!(
            "w的估计误差={}, b的估计误差={}",
            (&true_w.reshape(w.shape())? - w.as_tensor())?,
            (Tensor::new(&[true_b], device)?.reshape(b.shape()) - b.as_tensor())?
        )
    }

    Ok(())
}

/// 因在candle中提供了candle-nn crate来封装了模型
/// 故将candle_baisc_modle_dataset_noise_demo 自定义过程用candle-nn来实现
fn candle_basic_model_dataset_noise_nn_demo(device: &Device) -> Result<()> {
    // 线性模型
    let true_w = Tensor::new(&[2f32, 3.4], device)?;
    let true_b = 4.2f32;

    /// step-1: 生成数据集
    fn synthetic_data(
        w: &Tensor,
        b: f32,
        num_examples: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let X = Tensor::randn(0f32, 1.0f32, (num_examples, w.dim(0)?), &device)?;
        let y = (X.matmul(&w)? + b as f64)?;
        let y_clone = y.clone();
        let y = (y + Tensor::randn(0f32, 1.0f32, y_clone.shape(), &device)?)?;

        Ok((X, y.reshape(((), 1))?))
    }

    let (features, labels) = synthetic_data(&true_w.reshape(((), 1))?, true_b, 1000, device)?;
    println!("features={}, lables={}", features.i(0)?, labels.i(0)?);

    /// batch_size 小批量
    /// features 特征数据
    /// labels 标签数据
    fn data_iter(
        batch_size: usize,
        features: &Tensor,
        labels: &Tensor,
        device: &Device,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let num_examples = features.dim(0)?;
        let mut indices = Tensor::arange(0, num_examples as u32, &device)?.to_vec1::<u32>()?;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let mut batch_examples = Vec::<(Tensor, Tensor)>::new();

        for (i, batch_indices) in indices.chunks(batch_size).enumerate() {
            let batch_indices = Tensor::from_slice(batch_indices, (batch_size,), &device)?;
            let batch_features = features.i(&batch_indices)?;
            let batch_labels = labels.i(&batch_indices)?;
            // println!("batch_index={i}, batch_features={batch_features}, batch_labels={batch_labels}");
            //
            batch_examples.push((batch_features, batch_labels));
        }

        Ok(batch_examples)
    }
    let batch_size = 10;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = linear(2, 1, vb.pp("linear"))?;
    let params = ParamsAdamW {
        lr: 0.003,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    for step in 0..10 {
        for (batch_i, (sample_x, sample_y)) in data_iter(batch_size, &features, &labels, &device)?
            .into_iter()
            .enumerate()
        {
            let ys = model.forward(&sample_x)?;
            // 损失函数
            // let loss = (ys.sub(&sample_y)?.sqr()?/2f64)?.sum_all()?;
            let loss = (ys.sub(&sample_y)?.sqr()? / 2f64)?;
            // 反向传播 得到梯度最优的参数
            opt.backward_step(&loss)?;
            // println!("{step} {}", loss.sum_all()?.to_vec0::<f32>()?);

            // println!("{:?} loss={}, w={}, b={},lr={:?}"
            //          ,step+1
            //          , loss.sum_all()?.to_vec0::<f32>()?
            //          ,model.weight()
            //          , model.bias().unwrap()
            //          , opt.learning_rate()
            //          ,
            // );
        }

        // 测试
        // let train_1 = ((model.forward(&features)?).sub(&labels)?.sqr()?/2f64)?;
        // println!("迭代优化epoch {:?}, loss {:.2}", step+1, train_1.mean(0)?);
        // println!("w={}, b={},lr={:?}", model.weight()
        //          , model.bias().unwrap()
        //          , opt.learning_rate()
        //          ,
        // );
    }

    println!(
        "最终输出w={}, b={},lr={:?}",
        model.weight(),
        model.bias().unwrap(),
        opt.learning_rate(),
    );

    let w = model.weight();
    let b = model.bias().unwrap();
    println!(
        "w的估计误差={}, b的估计误差={}",
        (&true_w.reshape(w.shape())? - w)?,
        (Tensor::new(&[true_b], device)?.reshape(b.shape()) - b)?
    );

    Ok(())
}

/// 简单的线性回归模型，基于candle-nn实现
fn candle_basic_modle_sgd_linear_regression_demo(device: &Device) -> Result<()> {
    // 定义一参照组
    let true_w = Tensor::new(&[[3f32, 1.0]], device)?;
    let true_b = Tensor::new(-2f32, device)?;
    let true_m = Linear::new(true_w.clone(), Some(true_b.clone()));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = true_m.forward(&sample_xs)?;

    /// 接下来使用SGD优化参数的结果和实际进行比较
    /// w 和 b 置为0
    let train_w = Var::new(&[[0f32, 0.]], device)?;
    let train_b = Var::new(0f32, device)?;
    // 优化器
    let mut sgd = SGD::new(vec![train_w.clone(), train_b.clone()], 0.004)?;
    let train_m = Linear::new(
        train_w.as_tensor().clone(),
        Some(train_b.as_tensor().clone()),
    );
    // 迭代训练
    for _epoch in 0..1000 {
        // 预测
        let train_ys = train_m.forward(&sample_xs)?;
        // 损失函数
        let loss = (train_ys.sub(&sample_ys)?.sqr()? / 2f64)?.sum_all()?;
        // 优化
        //sgd.step(&loss.backward()?); // 同下
        sgd.backward_step(&loss)?;
    }
    println!(
        "最终输出w={}, b={},loss={} {}",
        train_w,
        train_b,
        (true_w.clone() - (&train_w).as_tensor())?,
        (true_b.clone() - (&train_b).as_tensor())?
    );

    Ok(())
}

/// 代码基本上同上，只是使用Adam优化器
fn candle_basic_modle_sgd_linear_regression_adm_demo(device: &Device) -> Result<()> {
    // 定义一参照组
    let true_w = Tensor::new(&[[3f32, 1.0]], device)?;
    let true_b = Tensor::new(-2f32, device)?;
    let true_m = Linear::new(true_w.clone(), Some(true_b.clone()));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = true_m.forward(&sample_xs)?;

    /// 接下来使用ADM优化参数的结果和实际进行比较
    /// w 和 b 置为0
    let train_w = Var::new(&[[0f32, 0.]], device)?;
    let train_b = Var::new(0f32, device)?;
    // 有时候可能存在将w和b设置为其他值的情况，所以需要使用VarMap来管理变量
    // 定义命名变量
    let mut var_map = VarMap::new();
    let train_w = var_map.get((1, 2), "w", Const(0.), DType::F32, &Device::Cpu)?;
    let train_b = var_map.get((), "b", Const(0.), DType::F32, &Device::Cpu)?;
    let train_c = var_map.get((), "c", Const(0.), DType::F32, &Device::Cpu)?;
    // #[derive(Clone, Debug)]
    // pub struct SimpleLinear {
    //     weight: Tensor,
    //     bias: Option<Tensor>,
    //     noise: Option<Tensor>,
    // }
    //
    // impl SimpleLinear {
    //     pub fn new(weight: Tensor, bias: Option<Tensor>, noise: Option<Tensor>) -> Self {
    //         Self { weight, bias, noise }
    //     }
    //
    //     pub fn weight(&self) -> &Tensor {
    //         &self.weight
    //     }
    //
    //     pub fn bias(&self) -> Option<&Tensor> {
    //         self.bias.as_ref()
    //     }
    //
    //     pub fn noise(&self) -> Option<&Tensor> {
    //         self.noise.as_ref()
    //     }
    // }
    //
    // /// 实现Module
    // impl Module for SimpleLinear {
    //     fn forward(&self, x: &Tensor) -> Result<Tensor> {
    //         let w = match *x.dims() {
    //             [b1, b2, _, _] => self.weight.broadcast_left((b1, b2))?.t()?,
    //             [bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
    //             _ => self.weight.t()?,
    //         };
    //         let x = x.matmul(&w)?;
    //         match &self.bias {
    //             None => Ok(x),
    //             Some(bias) => {
    //                 let x = x.broadcast_add(bias)?;
    //                 match &self.noise {
    //                     None => Ok(x),
    //                     Some(noise) => x.broadcast_add(noise),
    //                 }
    //             },
    //         }
    //     }
    // }

    // 优化器
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    // let mut opt = AdamW::new(vec![train_w.clone(), train_b.clone()], params)?;
    let mut opt = AdamW::new(var_map.all_vars(), params)?;
    let train_m = Linear::new(train_w.clone(), Some(train_b.clone()));
    // let train_m = SimpleLinear::new(train_w.clone(), Some(train_b.clone()), Some(train_c.clone()));

    // 迭代训练
    for _epoch in 0..1000 {
        // 预测
        let train_ys = train_m.forward(&sample_xs)?;
        // 损失函数
        let loss = (train_ys.sub(&sample_ys)?.sqr()? / 2f64)?.sum_all()?;
        // 优化
        //sgd.step(&loss.backward()?); // 同下
        opt.backward_step(&loss)?;
    }
    // println!("最终输出w={}, b={},loss={} {}"
    //          , train_w
    //          , train_b
    //          , (true_w.clone()-(&train_w).as_tensor())?
    //          , (true_b.clone()-(&train_b).as_tensor())?);

    // println!("最终输出w={}, b={},loss={} {}"
    //          , train_w
    //          , train_b
    //          , (true_w.clone()-(&train_w))?
    //          , (true_b.clone()-(&train_b))?);

    println!(
        "输出w={}, b={}, c={},loss={} {}",
        train_w,
        train_b,
        train_c,
        (true_w.clone() - (&train_w))?,
        (true_b.clone() - (&train_b))?
    );
    // 重置w和b
    var_map.set([("w", Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)].into_iter())?;
    var_map.set([("b", Tensor::ones((), DType::F32, &Device::Cpu)?)].into_iter())?;
    println!(
        "重置后输出w={}, b={},loss={} {}",
        train_w,
        train_b,
        (true_w.clone() - (&train_w))?,
        (true_b.clone() - (&train_b))?
    );
    Ok(())
}

/// softmax
fn candle_basic_model_ops_demo(device: &Device) -> Result<()> {
    let datas = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(datas, device)?;
    let t0 = softmax(&tensor.log()?, 0)?;
    let t1 = softmax(&tensor.log()?, 1)?;
    let t2 = softmax(&tensor.log()?, 2)?;
    println!("t0= {t0}, t1={t1}, t2={t2}");

    Ok(())
}

/// 图像分类集
/// 注意：添加hf-hub依赖
/// cargo add hf-hub[同步]或cargo add hf-hub --features tokio[异步]
/// 可能会因为“墙”的原因，需要手动下载mnnist数据集
/// [huggingface_datasets_mnnist](https://huggingface.co/datasets/ylecun/mnist/tree/refs%2Fconvert%2Fparquet/mnist)
fn load_parquet(parquet: SerializedFileReader<std::fs::File>) -> Result<(Tensor, Tensor)> {
    let samples = parquet.metadata().file_metadata().num_rows() as usize;
    let mut buffer_images: Vec<u8> = Vec::with_capacity(samples * 784);
    let mut buffer_labels: Vec<u8> = Vec::with_capacity(samples);
    for row in parquet.into_iter().flatten() {
        for (_name, field) in row.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                buffer_labels.push(*label as u8);
            }
        }
    }
    let images = (Tensor::from_vec(buffer_images, (samples, 784), &Device::Cpu)?
        .to_dtype(DType::F32)?
        / 255.)?;
    let labels = Tensor::from_vec(buffer_labels, (samples,), &Device::Cpu)?;
    Ok((images, labels))
}
fn load_mnist_images() -> Result<candle_datasets::vision::Dataset> {
    use parquet::file::reader::SerializedFileReader;

    let test_parquet_filename = "fashion_mnist/test/00001.parquet";
    let train_parquet_filename = "fashion_mnist/train/00001.parquet";
    let test_parquet = SerializedFileReader::new(std::fs::File::open(test_parquet_filename)?)
        .map_err(|e| {
            println!("{e}");
            Error::Msg(format!("Parquet error: {e}"))
        })?;
    let train_parquet = SerializedFileReader::new(std::fs::File::open(train_parquet_filename)?)
        .map_err(|e| Error::Msg(format!("Parquet error: {e}")))?;
    let (test_images, test_labels) = load_parquet(test_parquet)?;
    let (train_images, train_labels) = load_parquet(train_parquet)?;
    Ok(candle_datasets::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    })
}
/// batch_size 小批量
/// features 特征数据
/// labels 标签数据
fn data_iter (
    batch_size: usize,
    features: &Tensor,
    labels: &Tensor,
    device: &Device,
    num_cpus: Option<usize>,
) -> Result<Vec<(Tensor, Tensor)>>  {
    let num_examples = features.dim(0)?;
    let mut indices = Tensor::arange(0, num_examples as u32, &device)?.to_vec1::<u32>()?;
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    let mut batch_examples = Vec::<(Tensor, Tensor)>::new();

    let mut shape_dim0_size = batch_size;

    for (i, batch_indices) in indices.chunks(batch_size).enumerate() {
        // 如下代码是为了防止按照batch_size来chunk数据时，剩下的部分不足batch_size，否则无法在构建Tensor满足shape的要求
        if batch_indices.len() < batch_size {
            shape_dim0_size = batch_indices.len();
        }

        let temp_batch_indices = Tensor::from_slice(batch_indices, (shape_dim0_size,), &device)?;
        let batch_features = features.i(&temp_batch_indices)?;
        let batch_labels = labels.i(&temp_batch_indices)?;
        // println!("batch_index={i}, batch_features={batch_features}, batch_labels={batch_labels}");
        //
        batch_examples.push((batch_features, batch_labels));
    }

    Ok(batch_examples)
}

const LABEL_MAGIC_NO: u32 = 2049;
const IMG_MAGIC_NO: u32 = 2051;
/// ===============================
///  读取label相关
/// ===============================
fn extract_labels(decoder: &mut GzDecoder<File>) -> io::Result<([u32; 2], Vec<u8>)> {
    let mut metadata_buf = [0u32; 2];

    let mut buf_reader = BufReader::new(decoder);
    buf_reader.read_u32_into::<BigEndian>(&mut metadata_buf)?;

    let mut labels = Vec::new();
    buf_reader.read_to_end(&mut labels)?;
    if metadata_buf[0] != LABEL_MAGIC_NO {
        Err(io::Error::new(ErrorKind::InvalidData,
                           "Unable to verify FashionMNIST label data. Force redownload."))
    } else {
        Ok((metadata_buf, labels))
    }
}

/// ===============================
/// 读取图片相关
/// ===============================
fn extract_images(decoder: &mut GzDecoder<File>) -> io::Result<([u32; 4], Vec<Vec<u8>>)> {
    let mut metadata_buf = [0u32; 4];
    let mut buf_reader = BufReader::new(decoder);

    buf_reader.read_u32_into::<BigEndian>(&mut metadata_buf)?;

    let mut imgs = Vec::new();
    buf_reader.read_to_end(&mut imgs)?;
    if metadata_buf[0] != IMG_MAGIC_NO {
        Err(io::Error::new(ErrorKind::InvalidData,
                       "Unable to verify FashionMNIST images data. Force redownload."))
    } else {
        Ok((metadata_buf, imgs.chunks(784).map(|x| x.into()).collect()))
    }

}
fn load_fashion_gz_images(device: &Device) -> Result<candle_datasets::vision::Dataset> {
    /// 读取fashion mnist的gzip文件
    let dir = std::path::Path::new("fashion_mnist");

    let mut file = File::open(&dir.join("test_gz/t10k-images-idx3-ubyte.gz"))?;
    let mut decoder = GzDecoder::new(file);
    let (test_images_metadata, test_images) = extract_images(&mut decoder)?;
    println!("test images metadata: {:?}", test_images_metadata);

    file = File::open(&dir.join("test_gz/t10k-labels-idx1-ubyte.gz"))?;
    decoder = GzDecoder::new(file);
    let (test_labels_metadata, test_labels) = extract_labels(&mut decoder)?;
    println!("test label metadata:{:?}", test_labels_metadata);

    file = File::open(&dir.join("train_gz/train-images-idx3-ubyte.gz"))?;
    decoder = GzDecoder::new(file);
    let (train_images_metadata, train_images) = extract_images(&mut decoder)?;
    println!("train images metadat: {:?}", train_images_metadata);

    file = File::open(&dir.join("train_gz/train-labels-idx1-ubyte.gz"))?;
    decoder = GzDecoder::new(file);
    let (train_labels_metadata, train_labels) = extract_labels(&mut decoder)?;
    println!("train label metadata:{:?}", train_labels_metadata);

    Ok(candle_datasets::vision::Dataset {
        train_images: Tensor::new(train_images,device)?.reshape((60000, 28, 28))?,
        train_labels: Tensor::new(train_labels,device)?.reshape((60000,))?,
        test_images: Tensor::new(test_images,device)?.reshape((10000, 28, 28))?,
        test_labels: Tensor::new(test_labels,device)?.reshape((10000,))?,
        labels: 10,
    })
}

fn get_fashion_mnist_labels(labels: &Tensor) -> Result<(Vec<&str>)> {
    let labels_tag = vec![
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ];

    let labels_tag = labels
        .to_vec1::<u8>()?
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let label = labels_tag.get(i);
            let res = match labels_tag.get(i) {
                Some(label)=> *label,
                None => "",
            };

            res
        })
        .collect::<Vec<_>>()
    ;

    Ok(labels_tag)
}
fn candle_basic_model_mnist_images_demo(device: &Device) -> Result<()> {
    // 下载图像数据集
    /// 通过使用candle-datasets下载数据集
    // let dataset = candle_datasets::vision::mnist::load()?;
    /// 先将数据集保存到本地，通过本地直接读取
    // let dataset = load_mnist_images()?;
    let dataset = load_fashion_gz_images(device)?;
    /// test数据集
    let test_images = dataset.test_images;
    let test_labels = dataset.test_labels;
    // println!("test_images={test_images}, test_labels={test_labels}");

    /// train数据集
    let train_images = dataset.train_images;
    let train_labels = dataset.train_labels;
    // println!("train_images={train_images}, train_labels={train_labels}");
    println!(
        "images: test_dims={:?},test_dim0={:?}, train_dims={:?},train_dim0={:?}; \
         labels: test_labels_shape={:?}, train_labels_shape={:?}",
        test_images.dims(),
        test_images.dim(0)?,
        train_images.dims(),
        train_images.dim(0)?,
        test_labels.shape() ,
        train_labels.shape()
    );

    /// =========================================================================
    /// TODO：如下绘图代码有些bug，只是为了简单的验证
    /// =========================================================================
    const OUT_FILE_NAME: &str = "label_tag_blit-bitmap.png";
    // use plotters::prelude::*;

    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    root.split_evenly((2,9));

    let mut chart = ChartBuilder::on(&root)
        .caption("Bitmap Example", ("sans-serif", 30))
        .margin(5)
        .build_cartesian_2d(0.0..280.0, 0.0..280.0).unwrap();

    chart.configure_mesh().disable_mesh().draw().unwrap();
    let (w, h) = chart.plotting_area().dim_in_pixel();

    /// 小批量数据
    for (batch_i, (train_images_x, train_labels_y)) in data_iter(18, &train_images, &train_labels, device,Some(10))?.into_iter().enumerate()
    {
        // println!("batch_i={batch_i}, train_images_x={:?}, train_labels_y={:?}",(&train_images_x.as_ref()).shape(),(&train_labels_y.as_ref()).shape());
        let _test_labels_tag = get_fashion_mnist_labels(&train_labels_y)?;
        // let _test_labels_tag = get_fashion_mnist_labels(&train_labels_y)?.into_iter().filter(|x| !x.is_empty()).collect::<Vec<_>>();
        // println!("label tags ={:?}", _test_labels_tag);
        let i = batch_i.to_owned();
        let test_image_datas = (&train_images_x);
        // println!("test image datas shape={:?},datas={}", test_image_datas.shape(), test_image_datas);
        let w = test_image_datas.shape().dim(1)? as u32;
        let h = test_image_datas.shape().dim(2)? as u32;
        println!("test image datas shape={:?},w={:?}, h={:?}", test_image_datas.shape(), w, h);

        //let buffer = test_image_datas.i((row,))?.to_vec2::<u8>()?.into_iter().flatten().collect::<Vec<u8>>();
        let buffer = test_image_datas.to_vec3::<u8>()?.into_iter().flatten().flatten().collect::<Vec<u8>>();

        let grayimage = GrayImage::from_vec(w,h, buffer);
        let buffer = DynamicImage::from(grayimage.unwrap()).to_rgb8().to_vec();
        let bitmap  = BitMapElement::with_owned_buffer((0.05+w as f64, 0.95 + h as f64), (w, h), buffer);
        if let Some(elem) = bitmap {
            println!("drawing bitmap...");
            chart.draw_series(std::iter::once(elem)).unwrap();
        }
    }
    //
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

/// 如下方法是辅助读取fashion-mnist的gzip文件
/// 读取gzip文件时需要注意采用BigEndian(大端序)还是LittleEndian(小端序)
/// 否则数据读取出现非法不正确数据
fn read_u32<T: Read>(reader: &mut T) -> std::io::Result<u32> {
    use byteorder::ReadBytesExt;
    reader.read_u32::<byteorder::BigEndian>()
}

/// 检测文件magic number是否正常
fn check_magic_number<T: Read>(reader: &mut T, expected: u32) -> Result<()> {
    let magic_number = read_u32(reader)?;
    if magic_number != expected {
        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("incorrect magic number {magic_number} != {expected}"),
        ))?;
    }
    Ok(())
}
/// 读取fashion-mnist标签文件
fn read_labels(filename: &std::path::Path) -> Result<Tensor> {
    let labels: GzDecoder<File> = GzDecoder::new(File::open(&filename)?);
    let mut buf_reader = BufReader::new(labels);
    check_magic_number(&mut buf_reader, 2049)?;
    let samples = read_u32(&mut buf_reader)?;
    let mut data = vec![0u8; samples as usize];
    buf_reader.read_exact(&mut data)?;
    let samples = data.len();
    Tensor::from_vec(data, samples, &Device::Cpu)
}
/// 读取fashion-mnist图像文件
fn read_images(filename: &std::path::Path) -> Result<Tensor> {
    let images: GzDecoder<File> = GzDecoder::new(File::open(&filename)?);
    let mut buf_reader = BufReader::new(images);
    check_magic_number(&mut buf_reader, 2051)?;
    let samples = read_u32(&mut buf_reader)? as usize;
    let rows = read_u32(&mut buf_reader)? as usize;
    let cols = read_u32(&mut buf_reader)? as usize;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len];
    buf_reader.read_exact(&mut data)?;
    let tensor = Tensor::from_vec(data, (samples, rows * cols), &Device::Cpu)?;
    tensor.to_dtype(DType::F32)? / 255.
}
/// 读取fashion-mnist数据集
fn load_fashion_mnist() -> Result<Dataset> {
    let home_dir = std::path::Path::new("datas");
    let dataset = candle_datasets::vision::mnist::load_dir(home_dir)?;
    println!("train images shape={:?}; labels={:?}", dataset.train_images.shape(), dataset.train_labels.shape());
    println!("test images shape={:?}; labels={:?}", dataset.test_images.shape(), dataset.test_labels.shape());
    Ok(dataset)
}
/// 读取gzip格式文件
fn load_fashion_mnist_gz() -> Result<Dataset> {
    let home_dir = std::path::Path::new("fashion_mnist");
    let mut files = HashMap::with_capacity(4);
    files.insert("train", ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",]);
    files.insert("test", ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",]);

    // train 数据集
    let train_images_path = home_dir.join("train_gz").join("train-images-idx3-ubyte.gz");
    let train_label_path = home_dir.join("train_gz").join("train-labels-idx1-ubyte.gz");
    let train_images = read_images(&train_images_path)?;
    let train_labels = read_labels(&train_label_path)?;

    // test 数据集
    let test_images_path = home_dir.join("test_gz").join("t10k-images-idx3-ubyte.gz");
    let test_label_path = home_dir.join("test_gz").join("t10k-labels-idx1-ubyte.gz");
    let test_images = read_images(&test_images_path)?;
    let test_labels = read_labels(&test_label_path)?;

    let dataset = candle_datasets::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    };

    println!("train images shape={:?}; labels={:?}", dataset.train_images.shape(), dataset.train_labels.shape());
    println!("test images shape={:?}; labels={:?}", dataset.test_images.shape(), dataset.test_labels.shape());
    Ok(dataset)
}

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;
fn candle_basic_model_fashionmnist_demo(device: &Device) -> Result<()> {
    /// 训练时定义的参数
    struct TrainingArgs {
        learning_rate: f64,
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        epochs: usize,
    }
    // 训练时定义的参数(默认)
    let default_args = TrainingArgs {
        learning_rate: 1.0,
        weight: None,
        bias: None,
        epochs: 200,
    };

    /// 如下自定义模型
    fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
        let ws = vs.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
        let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
        Ok(Linear::new(ws, Some(bs)))
    }

    trait Model: Sized {
        fn new(vs: VarBuilder) -> Result<Self>;
        fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    }
    struct LinearModel {
        linear: Linear,
    }
    impl Model for LinearModel {
        fn new(vs: VarBuilder) -> Result<Self> {
            let linear = linear_z(IMAGE_DIM, LABELS, vs)?;
            Ok(Self { linear })
        }

        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            self.linear.forward(xs)
        }
    }
    fn train_mnist(dataset: Dataset, device: &Device, trainargs: TrainingArgs) -> Result<()> {
        let W = Tensor::randn(0.0f32, 0.01, (LABELS,LABELS), device)?;
        let b = Tensor::zeros((LABELS,), DType::F32, device)?;

        let train_images = dataset.train_images.to_device(device)?;
        let train_labels = dataset.train_labels;
        let train_labels = train_labels.to_dtype(DType::U32)?.to_device(device)?;

        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&mut varmap, DType::F32, device);
        let model = LinearModel::new(vs)?;
        if let Some (weight) = trainargs.weight{
            varmap.set([("w", W)].into_iter())?;
        }
        if let Some(bias) = trainargs.bias {
            varmap.set([("b", b)].into_iter())?;
        }
        let mut sgd = SGD::new(varmap.all_vars(), trainargs.learning_rate)?;

        let test_images = dataset.test_images.to_device(device)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(device)?;
        for epoch in 1..trainargs.epochs {
            let logits = model.forward(&train_images)?;
            let log_sm = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
            let loss = candle_nn::loss::nll(&log_sm, &train_labels)?;
            sgd.backward_step(&loss)?;

            let test_logits = model.forward(&test_images)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
            println!(
                "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
                loss.to_scalar::<f32>()?,
                100. * test_accuracy
            );
        }

        Ok(())
    }

    let dataset = load_fashion_mnist_gz()?;
    train_mnist(dataset, &device, default_args)
}

/// dive into deep learning - softmax回归的从零开始实现
fn candle_basic_model_fashionmnist_define_demo(device: &Device) -> Result<()> {
    // let dataset = load_fashion_mnist()?;
    let dataset = load_fashion_mnist_gz()?;
    let W = Tensor::randn(0.0f32, 0.01, (IMAGE_DIM,LABELS), device)?;
    let b = Tensor::zeros(LABELS, DType::F32, device)?;

    let epochs = 200;
    let batch_size = 256;
    let train_images = dataset.train_images.to_device(device)?;
    let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(device)?;

    let test_images = dataset.test_images.to_device(device)?;
    let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(device)?;

    fn softmax(xs: &Tensor) -> Result<Tensor> {
        let xs = xs.exp()?;
        let sum = xs.sum_keepdim(D::Minus1)?;
        xs.broadcast_div(&sum)
    }
    // 测试softmax
    let X = Tensor::randn(0f32, 1., (4,5), device)?;
    // let softmax_res = softmax(&train_images)?;
    // let softmax_res = softmax(&X)?;
    // println!("softmax: {}, sum={}", softmax_res, softmax_res.sum(1)?);

    fn net(xs: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        println!("xs shape={:?}, dims={:?}", xs.shape(), xs.dims());
        let dim = w.shape().dim(0)?;
        let xs = xs.reshape(((),dim))?;
        println!("after reshape: xs shape={:?}, dims={:?}", xs.shape(), xs.dims());
        let xs = xs.broadcast_matmul(&w)?;
        let xs = xs.broadcast_add(&b)?;
        softmax(xs.as_ref())
    }
    // 测试net
    let net_res = net(&train_images, &W, &b)?;
    // let net_res = net(&X, &W, &b)?;
    // println!("net= {}", net_res);

    fn cross_entropy(y_hat: &Tensor, y: &Tensor, device: &Device) -> Result<Tensor> {
        let temp_y = &y.reshape(((),1))?;
        println!("reshape temp_y: shape={:?}", temp_y.shape());
        let temp = y_hat.gather(temp_y, 1)?;
        println!("temp probability={}", temp);

        temp.log()? * -1.
    }

    // 测试cross_entropy
    let y = Tensor::new(&[0u32,2],device)?;
    let y_hat = Tensor::new(&[[0.1,0.3,0.6],[0.3,0.2,0.5]],device)?;
    // println!("cross_entropy(y_hat,y)={}", cross_entropy(&y_hat, &y)?);
    // println!("cross_entropy(y_hat,y)={}", cross_entropy(&net_res, &train_labels, device)?);
    // println!("cross_entropy(y_hat,y)={}", cross_entropy(&net_res, &y, device)?);
    fn accuracy(y_hat: &Tensor, y: &Tensor) -> Result<f64> {
        let dims = y_hat.shape().dims();
        let y_hat = if dims.len() > 1  && dims[1] > 1 {
            &(y_hat.argmax(D::Minus1)?)
        } else {
            y_hat
        };

        let cmp = y_hat.to_dtype(y.dtype())?.eq(y)?;

        let sum = cmp.to_dtype(y.dtype())?.sum_all()?.to_dtype(DType::F64)?;
        Ok(sum.to_scalar::<f64>()?)
    }

    // println!("accuracy(y_hat, y)/len(y)={}", accuracy(&net_res, &train_labels)?/((&train_labels).elem_count() as f64));
    /// 累加器
    #[derive(Debug, Clone)]
    pub struct Accumulator {
        data: Vec<f64>,
    }
    impl Accumulator {
        pub fn new(n: usize) -> Self {
            Accumulator {
                data: vec![0.0; n],
            }
        }

        pub fn add(&mut self, args: &[f64]) {
            if args.len() != self.data.len() {
                panic!("The number of arguments must match the length of the data vector.");
            }
            for (a, b) in self.data.iter_mut().zip(args) {
                *a += *b;
            }
        }

        pub fn get(&self, idx: usize) -> f64 {
            *self.data.get(idx).unwrap()
        }
        pub fn reset(&mut self) {
            for a in self.data.iter_mut() {
                *a = 0.0;
            }
        }


    }
    impl Index<usize> for Accumulator {
        type Output = f64;

        fn index(&self, idx: usize) -> &Self::Output {
            &self.data[idx]
        }
    }

    fn evaluate_accuracy(w: &Tensor, b: &Tensor, net: impl Fn(&Tensor, &Tensor,&Tensor) -> Result<Tensor>, batch: (Tensor, Tensor), device: &Device) -> Result<(f64,f64)> {
        let X = batch.0;
        let y = batch.1;
        let y_hat = net(&X, w, b).unwrap();
        let accuracy = accuracy(&y_hat, &y).unwrap();

        Ok((accuracy, y.elem_count() as f64))
    }

    // 测试evaluate_accuracy
    // 一次性测试
    // let evaluate_accuracy_res = evaluate_accuracy(&W, &b, net, (train_images.clone(), train_labels.clone()), device)?;
    // println!("evaluate_accuracy={:?}", evaluate_accuracy_res);
    // 按照256个数据一个批次进行测试
    // data_iter(batch_size, &train_images, &train_labels, device, None).iter().for_each(|batches| {
    //     batches.iter().for_each(|batch| {
    //         let (X, y) = batch;
    //         println!("X shape={:?}, y shape={:?}", X.shape(), y.shape());
    //         let accuracy = evaluate_accuracy(&W, &b, net, (X.clone(),y.clone()), device).unwrap();
    //         println!("accuracy={:?}", accuracy);
    //     });
    // });

    fn updater(batch_size: usize, w: Option<&Tensor>, b: Option<&Tensor>, lr: Option<f64>, device: &Device) -> Result<(SGD)>{
        let mut varmap = VarMap::new();
        if w.is_some() {
            varmap.set_one("w", &w.clone().unwrap())?;
            if b.is_some() {
                varmap.set_one("b", &b.clone().unwrap())?;
            }
        }

        let train_w = varmap.get((IMAGE_DIM,LABELS), "w", Const(0.), DType::F32, &Device::Cpu)?;
        let train_b = varmap.get((LABELS), "b", Const(0.), DType::F32, &Device::Cpu)?;
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut sgd = SGD::new(varmap.all_vars(), lr.unwrap_or(0.01));
        sgd
    }
    fn train_epoch_ch3(
        train_w: &Tensor,
        train_b: &Tensor,
        net: impl Fn(&Tensor, &Tensor,&Tensor) -> Result<Tensor>,
        train_iter: impl Iterator<Item = (Tensor, Tensor)>,
        loss: impl Fn(&Tensor, &Tensor, &Device) -> Result<Tensor>,
        is_nested: bool,
        updater: impl Fn(usize, Option<&Tensor>, Option<&Tensor>, Option<f64>, &Device) -> Result<SGD>,
        device: &Device,
    ) -> Result<(f64, f64)> {
        let mut metric = Accumulator::new(3);
        let vars = vec![Var::from_tensor(&train_w.clone())?, Var::from_tensor(&train_b.clone())?];
        for (X, y) in train_iter {
            let y_hat = net(&X, &train_w, &train_b)?;
            let l = loss(&y_hat, &y, device)?;
            if is_nested {
                let ll = l.mean_all()?.backward()?;
                let mut optimizer = candle_nn::SGD::new(vars.clone(), 0.01)?;
                optimizer.step(&ll)?
            } else {
                let ll = l.sum_all()?.backward()?;
                updater(1, Some(&train_w), Some(&train_b), Some(0.01), device)?.step(&ll)?;
            }

            let sum_all = l.sum_all()?.to_scalar::<f32>()? as f64;
            let accuracy = accuracy(&y_hat, &y)?;
            let numl = y.elem_count() as f64;
            metric.add(&[sum_all, accuracy, numl]);
        }

        Ok((metric.get(0)/metric.get(2), metric.get(1)/metric.get(2)))
    }
    pub(crate) fn train_ch3(
        train_w: &Tensor,
        train_b: &Tensor,
        num_epochs: usize,
        net: impl Fn(&Tensor, &Tensor,&Tensor) -> Result<Tensor>,
        train_iter: impl Iterator<Item = (Tensor, Tensor)>,
        test_iter: impl Iterator<Item = (Tensor, Tensor)>,
        loss: impl Fn(&Tensor, &Tensor, &Device) -> Result<Tensor>,
        updater: impl Fn(usize, Option<&Tensor>, Option<&Tensor>, Option<f64>, &Device) -> Result<SGD>,
        device: &Device,
    ) -> Result<()> {
        let mut train_metrics = (0.0, 0.0);
        let mut test_datas = test_iter.into_iter().map(|batch| {
            batch
        }).collect::<Vec<(Tensor, Tensor)>>();

        let train_datas = train_iter.into_iter().map(|batch| {
            batch
        }).collect::<Vec<(Tensor, Tensor)>>();

        for epoch in 0..num_epochs {
            println!("==={epoch}===");
            train_metrics = train_epoch_ch3(&train_w, &train_b, &net, train_datas.clone().into_iter(), &loss, true, &updater, device)?;
            let test_acc = evaluate_accuracy(&train_w, &train_b, &net, test_datas.clone().pop().unwrap(), device)?;

            println!("Epoch {:?}, train loss {:?}, train acc {:?}, test acc {:?}",(epoch+1), train_metrics.0, train_metrics.1, test_acc.0);
        }
        //
        println!("train loss {:?}, train acc {:?}", train_metrics.0, train_metrics.1);

        Ok(())
    }

    // 测试train_ch3
    let num_epochs = 10;
    let train_iter = data_iter(batch_size, &train_images, &train_labels, device, None)?;
    let test_iter = data_iter(batch_size, &test_images, &test_labels, device, None)?;

    // let trains = vec![(train_images, train_labels)];
    // let tests = vec![(test_images, test_labels)];

    train_ch3(&W, &b, num_epochs, net,
              train_iter.into_iter(),
              test_iter.into_iter(),
              cross_entropy, updater, device)?;

    Ok(())
}

fn candle_basic_model_fashionmnist_simple_demo(device: &Device) -> Result<()> {
    let dataset = load_fashion_mnist_gz()?;
    // 单个batch数据量
    let batch_size = 256;
    // train数据集
    let train_images = dataset.train_images;
    let train_labels = dataset.train_labels.to_device(&device)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&device)?;
    let train_iter = data_iter(batch_size, &train_images, &train_labels, device, None)?;

    // test数据集
    let test_images = dataset.test_images;
    let test_labels = dataset.test_labels;
    let test_labels = test_labels.to_dtype(DType::U32)?.to_device(&device)?;
    let test_iter = data_iter(batch_size, &test_images, &test_labels, device, None)?;

    let W = Tensor::randn(0.0f32, 0.01, (IMAGE_DIM,LABELS), device)?;
    let b = Tensor::zeros(LABELS, DType::F32, device)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap.clone(), DType::F32, &device);
    // varmap.set_one("w", W.clone())?;
    // varmap.set_one("b", b.clone())?;
    // varmap.set([("w", Tensor::randn(0f32, 0.01f32, (IMAGE_DIM, LABELS), &Device::Cpu)?)].into_iter())?;
    // varmap.set([("b", Tensor::ones((), DType::F32, &Device::Cpu)?)].into_iter())?;


    let net = candle_nn::sequential::seq()
        .add(linear::linear(IMAGE_DIM, LABELS, vb.clone())?)
        ;
    fn type_of<T>(_: T) -> &'static str {
        std::any::type_name::<T>()
    }
    /// pytorch -> candle
    ///
    use candle::{Module, Tensor};
    use candle_nn::{VarBuilder, Linear};

    // 自定义Flatten结构体
    #[derive(Debug)]
    pub struct Flatten;
    impl Flatten {
        pub fn new() -> Self {
            Self
        }
    }
    impl Module for Flatten {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            // // 保持batch维度不变，展平其他维度
            // let dims = x.dims();
            // if dims.len() < 2 {
            //     return Err(Error::Msg("Input tensor must have at least 2 dimensions".into()));
            // }
            // let flattened_size: usize = dims[1..].iter().product();
            // x.reshape((dims[0], flattened_size))
            x.flatten(1,1)
        }
    }
    struct Model {
        flatten: Flatten,
        linear: Linear,
    }
    impl Model {
        fn new(vb: VarBuilder) -> Result<Self> {
            let linear = candle_nn::linear(
                784,
                10,
                vb,
            )?;
            Ok(Self {
                flatten: Flatten::new(),
                linear,
            })
        }
    }
    impl Module for Model {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let x = self.flatten.forward(x)?;
            self.linear.forward(&x)
        }
    }

    let net = candle_nn::sequential::seq()
        .add(Flatten::new())
        .add(linear::Linear::new(W, Some(b)))
        .add_fn(|x| {
            let normal = Tensor::randn_like(x, 0.0f64, 0.01f64);
            normal
        })
        ;
    // 使用示例
    // let device = Device::Cpu;
    let model = Model::new(vb.clone())?;
    let mut trainer = candle_nn::optim::SGD::new(varmap.to_owned().all_vars(),0.1f64)?;
    let num_epochs = 10;

    for _step in 0..num_epochs {
        println!("==={}===", _step);
        let ys = model.forward(&train_images)?;
        let loss = candle_nn::loss::cross_entropy(&ys, &train_labels)?;
        trainer.backward_step(&loss)?;

        let test_ys = model.forward(&test_images)?;
        let sum_ok = test_ys
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!("test accuracy: {:?}", test_accuracy);
    }
    println!("{:?}", &varmap.clone().all_vars());

    Ok(())
}


fn main() {
    let device = Device::Cpu;
    candle_basic_model_fashionmnist_simple_demo(&device);
    // candle_basic_model_fashionmnist_define_demo(&device);
    // candle_basic_model_fashionmnist_demo(&device);
    // candle_basic_model_mnist_images_demo(&device);
    //candle_basic_model_ops_demo(&device);
    // candle_basic_modle_sgd_linear_regression_adm_demo(&device);
    // candle_basic_modle_sgd_linear_regression_demo(&device);
    // candle_basic_model_dataset_noise_nn_demo(&device);
    // candle_baisc_modle_dataset_noise_demo(&device);
    // candle_basic_modle_linermodel_demo_v3(&device);
    // candle_basic_modle_linermodel_demo_v2(&device);
    // candle_basic_modle_linermodel_demo_v1(&device);
    // candle_basic_model_demo(&device);
}
