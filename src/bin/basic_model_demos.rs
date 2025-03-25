use candle_core::{DType, Device, Module, Result, Tensor};
use plotters::prelude::{BitMapBackend, ChartBuilder, Color, DiscreteRanged, IntoDrawingArea, IntoLinspace, IntoSegmentedCoord, LabelAreaPosition, LineSeries, PathElement, BLACK, BLUE, GREEN, RED, WHITE};
use plotters::prelude::full_palette::PURPLE;

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

fn main() {
    let device = Device::Cpu;
    candle_basic_modle_linermodel_demo_v3(&device);
    // candle_basic_modle_linermodel_demo_v2(&device);
    // candle_basic_modle_linermodel_demo_v1(&device);
    // candle_basic_model_demo(&device);
}