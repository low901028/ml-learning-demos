use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, Var};
use candle_core::backprop::GradStore;
use candle_nn::{linear, AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap, SGD};
use candle_nn::Init::Const;
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

/// 因在candle中提供了candle-nn crate来封装了模型
/// 故将candle_baisc_modle_dataset_noise_demo 自定义过程用candle-nn来实现
fn candle_basic_model_dataset_noise_nn_demo(device: &Device) -> Result<()> {
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

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = linear(2, 1, vb.pp("linear"))?;
    let params = ParamsAdamW {
        lr: 0.003,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    for step in 0..10 {
        for (batch_i, (sample_x, sample_y)) in data_iter(batch_size, &features, &labels, &device)?.into_iter().enumerate() {
            let ys = model.forward(&sample_x)?;
            // 损失函数
            // let loss = (ys.sub(&sample_y)?.sqr()?/2f64)?.sum_all()?;
            let loss = (ys.sub(&sample_y)?.sqr()?/2f64)?;
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

    println!("最终输出w={}, b={},lr={:?}", model.weight()
             , model.bias().unwrap()
             , opt.learning_rate()
             ,
    );

    let w = model.weight();
    let b = model.bias().unwrap();
    println!("w的估计误差={}, b的估计误差={}", (&true_w.reshape(w.shape())?-w)?, (Tensor::new(&[true_b], device)?.reshape(b.shape())-b)?);

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
    let train_w = Var::new(&[[0f32, 0.]],device)?;
    let train_b = Var::new(0f32,device)?;
    // 优化器
    let mut sgd = SGD::new(vec![train_w.clone(), train_b.clone()], 0.004)?;
    let train_m = Linear::new(train_w.as_tensor().clone(), Some(train_b.as_tensor().clone()));
    // 迭代训练
    for _epoch in 0..1000 {
        // 预测
        let train_ys = train_m.forward(&sample_xs)?;
        // 损失函数
        let loss = (train_ys.sub(&sample_ys)?.sqr()?/2f64)?.sum_all()?;
        // 优化
        //sgd.step(&loss.backward()?); // 同下
        sgd.backward_step(&loss)?;
    }
    println!("最终输出w={}, b={},loss={} {}"
             , train_w
             , train_b
             , (true_w.clone()-(&train_w).as_tensor())?
             , (true_b.clone()-(&train_b).as_tensor())?);

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
    let train_w = Var::new(&[[0f32, 0.]],device)?;
    let train_b = Var::new(0f32,device)?;
    // 有时候可能存在将w和b设置为其他值的情况，所以需要使用VarMap来管理变量
    // 定义命名变量
    let mut var_map = candle_nn::VarMap::new();
    let train_w = var_map.get((1, 2), "w", Const(0.), DType::F32, &Device::Cpu)?;
    let train_b = var_map.get((), "b", Const(0.), DType::F32, &Device::Cpu)?;

    // 优化器
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    // let mut opt = AdamW::new(vec![train_w.clone(), train_b.clone()], params)?;
    let mut opt = AdamW::new(var_map.all_vars(), params)?;
    let train_m = Linear::new(train_w.clone(), Some(train_b.clone()));
    // 迭代训练
    for _epoch in 0..1000 {
        // 预测
        let train_ys = train_m.forward(&sample_xs)?;
        // 损失函数
        let loss = (train_ys.sub(&sample_ys)?.sqr()?/2f64)?.sum_all()?;
        // 优化
        //sgd.step(&loss.backward()?); // 同下
        opt.backward_step(&loss)?;
    }
    // println!("最终输出w={}, b={},loss={} {}"
    //          , train_w
    //          , train_b
    //          , (true_w.clone()-(&train_w).as_tensor())?
    //          , (true_b.clone()-(&train_b).as_tensor())?);

    println!("最终输出w={}, b={},loss={} {}"
             , train_w
             , train_b
             , (true_w.clone()-(&train_w))?
             , (true_b.clone()-(&train_b))?);
    // 重置w和b
    var_map.set([("w", Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)].into_iter())?;
    var_map.set([("b", Tensor::ones((), DType::F32, &Device::Cpu)?)].into_iter())?;
    println!("重置后输出w={}, b={},loss={} {}"
             , train_w
             , train_b
             , (true_w.clone()-(&train_w))?
             , (true_b.clone()-(&train_b))?);
    Ok(())
}
fn main() {
    let device = Device::Cpu;
    candle_basic_modle_sgd_linear_regression_adm_demo(&device);
    // candle_basic_modle_sgd_linear_regression_demo(&device);
    // candle_basic_model_dataset_noise_nn_demo(&device);
    // candle_baisc_modle_dataset_noise_demo(&device);
    // candle_basic_modle_linermodel_demo_v3(&device);
    // candle_basic_modle_linermodel_demo_v2(&device);
    // candle_basic_modle_linermodel_demo_v1(&device);
    // candle_basic_model_demo(&device);
}