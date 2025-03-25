use std::ops::Div;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, Var};
use candle_core::backprop::GradStore;
use plotters::prelude::{BitMapBackend, ChartBuilder, Circle, Color, DiscreteRanged, EmptyElement, IntoDrawingArea, IntoFont, IntoLinspace, IntoSegmentedCoord, LabelAreaPosition, LineSeries, PathElement, PointSeries, Text, BLACK, BLUE, GREEN, RED, WHITE};
use plotters::prelude::full_palette::PURPLE;
use rand::prelude::SliceRandom;

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
        p * f32::exp(-0.5 / sigma.powf(2.0) * (x-mu).powf(2.0))
    }

    let x = 10.0f32;
    let mu = 0.0f32;
    let sigma = 1.0f32;
    println!("p(x={}) = {}", x, noraml(x, mu, sigma));

    let OUT_FILE_NAME = "basic_model_demos.png";
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (1376, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let root_area = root_area
        .titled("线性回归", ("sans-serif", 20)).unwrap();

    let x_axis = (-7.0f32..7.0).step(0.01);
    let params = [(0,1), (0,2), (3,1)];

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
        .build_cartesian_2d(-7.0f32..7.0f32,0.0f32..0.4).unwrap()
       ;

    chart
        .configure_mesh()
        // .disable_x_mesh()
        // .disable_y_mesh()
        .x_labels(20)
        .y_labels(20)
        .y_label_formatter(&|y| format!("{:.1}f", *y))
        .y_desc("P(x)")
        .x_desc("x")
        .draw().unwrap();

    chart.draw_series(LineSeries::new(x_axis.values()
                                       .map(|x| {
                                           (x, noraml(x, params.get(0).unwrap().0 as f32, params.get(0).unwrap().1 as f32))
                                       }), &BLUE)).unwrap()
        .label("mean 0， std 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart.draw_series(LineSeries::new(x_axis.values()
                                       .map(|x| {
                                           (x, noraml(x, params.get(1).unwrap().0 as f32, params.get(1).unwrap().1 as f32))
                                       }), &PURPLE)).unwrap()
        .label("mean 0, std 2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PURPLE));

    chart.draw_series(LineSeries::new(x_axis.values()
                                       .map(|x| {
                                           (x, noraml(x, params.get(2).unwrap().0 as f32, params.get(2).unwrap().1 as f32))
                                       }), &GREEN)).unwrap()
        .label("mean 3, std 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));


    // 配置边框
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();

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
    let root_area = root
        .titled("线性模型", ("sans-serif", 20)).unwrap();

    let mut chart = ChartBuilder::on(&root_area)
        .margin(5)
        .caption("散点图", ("sans-serif", 20))
        .set_label_area_size(LabelAreaPosition::Top, 60)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .build_cartesian_2d(-4.0f32..4.0f32,-10.0f32..20.0).unwrap()
        ;

    chart.configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .y_label_formatter(&|y| format!("{:.1}f", *y))
        .draw().unwrap();

    chart.draw_series(PointSeries::<_, _, Circle<_,_>, _>::new(
        // 特征和标签
        // 基于zip合并，构建元组(feature, label)
        (&features.i((..,1))?).to_vec1::<f32>()?.into_iter()
            .zip((&labels.i((..,0))?).to_vec1::<f32>()?.into_iter())
            // .enumerate()
            // .map(|(i, (feature, label))| {
            //     // println!("index={i}, feature={feature}, label={label}");
            //     (feature, label)
            // })
            .map(|(feature, label)| {
                // println!("feature={feature}, label={label}");
                (feature, label)
            })
            .collect::<Vec<_>>()
        // 先获取特征数据，再根据索引获取标签
        // (&features.i((..,1))?).to_vec1::<f32>()?.into_iter()
        //     .enumerate()
        //     .map(|(i, feature)| {
        //        let binding = (&labels.i((..,0)).unwrap()).to_vec1::<f32>().unwrap();
        //        let label= binding.get(i).unwrap().to_owned();
        //        // println!("feature={feature}, label={label}");
        //        (feature, label)
        //     })
        ,2
        , GREEN.filled(),
    )
    ).unwrap()
    ;
    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}
/// 根据带有噪声的线性来构建数据集
/// y = Xw + b + e
fn candle_baisc_modle_dataset_noise_demo(device: &Device) -> Result<()> {
    // 线性模型
    let true_w = Tensor::new(&[2f32,3.4], device)?;
    let true_b = 4.2f32;

    /// step-1: 生成数据集
    fn synthetic_data(w: &Tensor, b: f32, num_examples: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let X = Tensor::randn(0f32, 1.0f32, (num_examples, w.dim(0)?), &device)?;
        let y = (X.matmul(&w)? + b as f64)?;
        let y_clone = y.clone();
        let y = (y + Tensor::randn(0f32, 1.0f32, y_clone.shape(), &device)?)?;

        Ok( (X, y.reshape(((),1))?))
    }

    let (features, labels) = synthetic_data(&true_w.reshape(((),1))?, true_b, 1000, device)?;
    println!("features={}, lables={}", features.i(0)?, labels.i(0)?);
    // 绘制散点图
    // dataset_noise_draw_points_demo(&features, &labels);
    /// step-2 将生成的数据集打乱，并以小批量的形式返回
    /// batch_size 小批量
    /// features 特征数据
    /// labels 标签数据
    fn data_iter(batch_size: usize, features: &Tensor, labels: &Tensor, device: &Device) -> Result<Vec<(Tensor, Tensor)>>{
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
    // data_iter(batch_size, &features, &labels, device);
    /// step-3 初始化模型参数 由于需要grad，则需要Var来定义
    let w = Var::randn(0f32, 0.01f32, (2,1), &device)?;
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
        let l = (y_hat - y.reshape(y_hat.shape()))?.sqr()?/2f64;
        l
    }

    /// step-6 定义优化算法
    fn sgd(gradstore: GradStore, params: &[&Var], lr: f32, batch_size: usize) -> Result<()> {
        for param in params {
            let new_param = (param.as_tensor() - ((lr as f64) * gradstore.get(param.as_tensor()).expect(format!("no grad for {}", param).as_ref()))? / batch_size as f64)?;
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
        for (batch_i, (X, y)) in data_iter(batch_size, &features, &labels, &device)?.into_iter().enumerate() {
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
        println!("epoch {:?}, loss {:.2}", epoch+1, train_1.mean(0)?);
        // 通过训练得到相对较优的参数w和b
        println!("w={}, b={}", w.as_tensor(), b.as_tensor());
        println!("w的估计误差={}, b的估计误差={}", (&true_w.reshape(w.shape())?-w.as_tensor())?, (Tensor::new(&[true_b], device)?.reshape(b.shape())-b.as_tensor())?)
    }

    Ok(())
}


fn main() {
    let device = Device::Cpu;
    candle_baisc_modle_dataset_noise_demo(&device);
    // candle_basic_modle_linermodel_demo_v3(&device);
    // candle_basic_modle_linermodel_demo_v2(&device);
    // candle_basic_modle_linermodel_demo_v1(&device);
    // candle_basic_model_demo(&device);
}