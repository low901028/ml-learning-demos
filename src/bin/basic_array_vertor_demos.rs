// #![allow(dead_code, unreachable_code)]

extern crate core;
use candle_core::{DType, Device, IndexOp, Result, Tensor, Var, D};
use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ops::Mul;
use std::panic;
use std::path::PathBuf;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::generation::Sampling;
use plotters::prelude::{ChartBuilder, Color, IntoFont, LineSeries, PathElement, SVGBackend, BLACK, BLUE, RED, WHITE};
use plotters::drawing::IntoDrawingArea;
use log::log;

/// 数据处理
fn candle_datas_process_demo(device: &Device) -> Result<()> {
    /// 构建指定的范围生成Tensor，等同于pytorch.arange(12)
    /// 不过需要注意的再定义range的时候需要指定类型，具体参考[DType](https://github.com/candle/candle-core/src/dtype.rs)中定义的类型
    /// 输出： datas=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
    ///       Tensor[[12], u32]
    let datas = Tensor::arange(0u32, 12u32, device)?
            //.reshape((3,4))?
            //.to_dtype(candle_core::DType::U32)?
        ;
    println!("datas={}", datas);
    /// 获取Tensor的shape，等同pytorch中的shape
    println!("datas shape={:?}", datas.shape());
    /// 获取Tensor的元素个数，等同pytorch中的numel
    println!("datas the number of elements: {:?}", datas.elem_count());

    /// 改变tensor的shape，而不改变元素数量和元素值
    /// 需要注意的是 提供给reshape的参数里面最多可以提供一个()，并且shape不能指定负数
    ///  也可以按照需要提供明确的shape：shape(2,3), shape(2,2,3), shape(2,2,2,3)......诸如此类的shape都可以
    /// 确保指定的元素个数与指定shape中所有参数积乘一致，否则会报错
    ///
    /// 输出：[[ 0,  1,  2,  3],
    ///       [ 4,  5,  6,  7],
    ///       [ 8,  9, 10, 11]]
    ///     Tensor[[3, 4], u32]
    let datas = datas
        .reshape((3,4))?
        //.reshape((2,(),3))?
        ;
    println!("datas tensor reshape: {}", datas);

    /// 构建一个全0的Tensor，等同于pytorch.zeros(2,3,4)
    let datas = Tensor::zeros((2, 3, 4), DType::U32, &device)?;
    println!("datas zeros: {}", datas);

    /// 构建一个全1的Tensor，等同于pytorch.ones(2,3,4)
    let datas = Tensor::ones((2, 3, 4), DType::U32, &device)?;
    println!("datas ones: {}", datas);

    /// 通过从某个特定的概率分布中随机采样，构建一个Tensor，等同于pytorch.randn(2,3,4)
    /// 指定mean=0 std=1，shape=(2,3,4)的Tensor
    /// 需要注意提供的mean和std值必须提供类型，同时类型必须满足[FloatDType](https://github.com/huggingface/candle/blob/0b24f7f0a41d369942bfcadac3a3cf494167f8a6/candle-core/src/dtype.rs#L210)
    let datas = Tensor::randn(0f32, 1f32, (2, 3, 4), &device)?;
    println!("datas randn: {}", datas);

    /// 构建一个Tensor，等同于pytorch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
    /// 需要注意尽量在定义时指明元素类型
    let datas = //Tensor::new(&[[2,1,4,3],[1,2,3,4],[4,3,2,1]], &device)?
      Tensor::new(&[[2.0,1.,4.,3.],[1.0,2.,3.,4.],[4.,3.,2.,1.]], &device)?
        .to_dtype(DType::U32)?
        ;
    println!("datas new(): {}", datas);

    /// 接下来是Tensor的运算
    /// 输出内容：
    /// x+y=[ 3.,  4.,  6., 10.] Tensor[[4], f32];
    /// x-y=[-1.,  0.,  2.,  6.] Tensor[[4], f32],
    /// x*y=[ 2.,  4.,  8., 16.] Tensor[[4], f32],
    /// x/y=[0.5000, 1.0000, 2.0000, 4.0000] Tensor[[4], f32],
    /// x**y=[ 1.,  4., 16., 64.] Tensor[[4], f32]
    /// 需要注意进行运输需要确保的元素的类型一致
    let x = Tensor::new(&[1.0, 2., 4., 8.], &device)?.to_dtype(DType::F32)?;
    let y = Tensor::new(&[2u32, 2, 2, 2], &device)?.to_dtype(DType::F32)?;
    println!(
        "x+y={}; x-y={}, x*y={}, x/y={}, x**y={}",
        (&x + &y)?,
        (&x - &y)?,
        (&x * &y)?,
        (&x / &y)?,
        &x.pow(&y)?
    );
    // 等同上述的运算
    // println!("x+y={}; \nx-y={}; \nx*y={}; \nx/y={}; \nx**y={}",
    //     Tensor::add(&x, &y)?, Tensor::sub(&x, &y)?, Tensor::mul(&x, &y)?, Tensor::div(&x, &y)?, Tensor::pow(&x, &y)?
    // );

    /// 按元素求幂计算，等同于pytorch.exp(x)
    let datas = &x.exp()?
      //Tensor::exp(&x)?
        ;
    println!("datas exp= {}", datas);

    /// 矩阵拼接，等同于pytorch.cat([x,y],0)
    let X = Tensor::arange(0f32, 12f32, &device)?.reshape((3, 4))?;
    let Y = Tensor::new(
        &[
            [2.0f32, 1., 4., 3.],
            [1.0f32, 2., 3., 4.],
            [4.0f32, 3., 2., 1.],
        ],
        &device,
    )?;

    let XY = Tensor::cat(&[&X, &Y], 0)?;
    let YX = Tensor::cat(&[&X, &Y], 1)?;
    println!("[cat(X,Y) dim=0]={}, [cat(Y,X) dim=1]={}", XY, YX);

    println!("X == Y: {}, Sum(X)={}", (X.eq(&Y)?), X.sum((0, 1))?);

    /// 在candle中Tensor需要进行广播操作，是需要调用对应的带有broadcast_的方法的
    let a = Tensor::arange(0u32, 3u32, &device)?.reshape((3, 1))?;
    let b = Tensor::arange(0u32, 2u32, &device)?.reshape((1, 2))?;
    println!(
        "tensor boradcast:a={},b={}, a+b={}",
        &a,
        &b,
        &a.broadcast_add(&b)? // 不支持&a + &b
    );

    /// Tensor中的索引和切片
    /// 在candle中有多种基于index方式获取元素
    /// - index: 索引，比如x.i(1)表示获取第1行元素，
    ///             x.i((1,2))表示获取第1行第2列的元素，
    ///             x.i(2..)  表示获取第2行到最后行的元素，
    ///             x.i((1..3))表示获取第1行到第3行的元素[不包括第3行的元素]，
    ///             x.i((1..3,2))表示获取第1行第2列到第3列的第2列的元素，
    ///             x.i(indexs_slice_tensor) 表示获取indexs_slice_tensor索引集获取指定的元素；
    ///             tensor对应的是tensor的索引index集构成的Tensor
    let x = &X.i(1)?;
    println!("tensor x index=1:{}", x);
    let x = &X.i((1, 2))?;
    println!("tensor x index=[r=1,c=2]:{}", x);
    let x = &X.i(2..)?;
    println!("tensor x index=2..:{}", x);
    let x = &X.i(1..3)?;
    println!("tensor x index  range[1..3]={}", x);
    let x = &X.i((1..3, 2))?;
    println!("tensor x index  range[1..3,2]={}", x);
    let indexs = Tensor::new(&[1u32, 2], &device)?;
    let x = &X.i(&indexs)?;
    println!("tensor x index by tensor(slice)={}", x);

    /// 需要注意：在candle中Tensor是不能被修改的，只能通过其他方式产生新的Tensor来达到修改目的
    /// 如下示例：将tensor加上数[需要对应的数字类型必须为f64]，修改后的Tensor并不会影响原来的Tensor，会产生新的Tensor
    let x = (&X + 8f64)?;
    println!("X={}; index=(1,2)+8f64={}", X, x);

    /// 修改tensor中某一个位置的元素，需要使用slice_assign方法
    /// 先要基于最新值构建一个tensor，并指定shape【shape是需要根据被修改内容的范围来指定的】
    ///  - 基于ones*最新值(类型需要用f64)来构建根据tensor修改范围来定义修改值tensor的shape；可用于修改相同的值；
    ///  - 基于new来定义并指定最新值(只要能满足定义最新值的tensor即可，并不是只能使用new函数)，这样可以修改不同值；
    /// *** 需要注意这里所谓的修改Tensor，并不是赋值，而是替换，即修改了Tensor的内容，但是不会影响原来的Tensor ***
    // 最新值全部一样
    let to_insert = (Tensor::ones((3, 1), DType::F32, &device)? * 9f64)?;
    // 最新值并不一样
    let to_insert = Tensor::new(&[9f32, 19., 29.], &device)?.reshape((3, 1))?;
    let x = &X.slice_assign(&[0..3, 2..3], &to_insert)?;
    println!("source={}; update index=[0..3,2..3]={}", &X, x);
    // 修改tensor具体位置的内容
    let to_insert = (Tensor::ones((1, 1), DType::F32, &device)? * 999f64)?;
    let x = &X.slice_assign(&[1..2, 2..3], &to_insert)?;
    println!("source={}; update index=(1,2)={}", &X, x);

    /// 获取tensor的id，即tensor的内存地址
    /// Tensor Y在计算前后的tensorId是不一样的
    let before = &Y.id();
    let Y = (&Y + &X)?;
    let after = &Y.id();
    println!("Tensor Y id: before={:?}; after={:?}", before, after);

    let Z = Tensor::zeros_like(&Y)?;
    println!("Z id={:?}, Y id={:?}", &Z.id(), &Y.id());

    // 由于在candle中，提供了Ndarray;
    // 同时也提供了多种方式将tensor转vec[按照Tensor的shape来调用不同to_vecX()方法]
    // 若是比较需要将Tensor转Ndarray,可以使用narray crate来完成candle中Tensor和narray中Ndarray的转换
    let vec = &X.to_vec2::<f32>()?;
    println!("tensor to vec: {:?}", vec);

    let a = Tensor::new((3.5f32), &device)?;
    let a_var = Var::from_tensor(&a)?;
    println!("var a={}", a_var);
    let a_scaler = a_var.to_scalar::<f32>()?;
    println!("tensor a={} to scaler:{:?}", a, a_scaler);

    let b = Var::new(123u32, device)?;
    let b_scaler = b.to_scalar::<u32>()?;
    println!("var b={} to scaler:{:?}", b, b_scaler);

    /// 此处使用Var，在candle中Var是对Tensor的包装，不同于Tensor的不可变，Var是可以修改其中的内容
    /// 具体见另外的basic_var_demos.rs
    let ids = Tensor::new(&[0u32, 1u32, 1u32], device)?;
    let v_X = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    let init = Tensor::ones((4, 2), DType::F32, device)?;
    let v_X_b = init.index_add(&ids, &v_X, 1)?;
    println!("v_X_b={}, init={}, v_X={}", v_X_b, init, v_X);

    Ok(())
}

/// 数据集的读取
fn read_data_source_demo(device: &Device) -> Result<()> {
    #[derive(Debug, serde::Deserialize, serde::Serialize, Eq, PartialEq)]
    struct Record {
        NumRooms: String,
        Alley: String,
        Price: i32,
    }

    // 实现Display; 输出{}时的格式
    impl Display for Record {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "num-rooms={}, alley={}, price={}",
                self.NumRooms, self.Alley, self.Price
            )
        }
    }

    /// 1、构建数据
    let path: PathBuf = [r".", "data", "house_tiny.csv"].iter().collect();
    if !path.exists() {
        let parent = path.parent().unwrap();
        let result = std::fs::DirBuilder::new().recursive(true).create(parent);
        match result {
            Ok(_) => {
                println!("创建目录成功");
                if let mut f = OpenOptions::new().create(true).write(true).open(path)? {
                    f.write("NumRooms,Alley,Price\n".as_bytes());
                    f.write("NA,Pave,127500\n".as_bytes());
                    f.write("2,NA,106000\n".as_bytes());
                    f.write("4,NA,178100\n".as_bytes());
                    f.write("NA,NA,140000".as_bytes());
                }
            }
            Err(error) => {
                println!("创建目录失败: {}", error);
            }
        }
    } else {
        /// 2.读取数据
        if let Ok(file) = File::open(path) {
            let mut reader = csv::ReaderBuilder::new().from_reader(file);
            // 输出读取到的数据
            // for result in reader.deserialize() {
            //     let record: Record = result.unwrap();
            //     println!("{:?}", record);
            // }
            // 为Tensor提供源数据
            let prices: Vec<u32> = reader
                .deserialize::<Record>()
                .map(|result| match result {
                    Ok(r) => r.Price as u32,
                    Err(_) => 0u32,
                })
                .collect();
            let records_tensor = Tensor::from_iter(prices, device)?;
            println!("iter to tensor:{}", records_tensor);
        }

        // 如下示例也可以实现读取文件
        // if let file = File::open(path) {
        //     match file {
        //         Ok(file) => {
        //             println!("文件已存在");
        //             let mut reader = csv::ReaderBuilder::new().from_reader(file);
        //             for result in reader.deserialize() {
        //                 let record: Record = result.unwrap();
        //                 println!("{:?}", record);
        //             }
        //         },
        //         Err(error) => {
        //             println!("文件不存在: {}", error);
        //         }
        //     }
        // }
    }

    Ok(())
}

/// 异常数据修复
fn candle_datas_process_nan_demo(device: &Device) -> Result<()> {
    let datas = [
        [5., 6.0f32, f32::NAN],
        [f32::INFINITY, f32::NAN, 7.],
        [8.0f32, 6., 3.],
    ];

    let datas= datas
        .map(|d|
                d.iter()
                 .map(|x|
                     if x.is_nan() {
                         0.
                     } else if x.is_infinite() {
                         0.
                     } else {
                         *x
                     })
                    .collect::<Vec<_>>()
           );
    println!("datas = {:?}", &datas);
    Ok(())
}

/// candle在线性代数中应用
fn candle_liner_algebra_basic_demo(device: &Device) -> Result<()> {
    /// 矩阵的转置
    let B = Tensor::new(&[[0f32, 2., 1.], [2., 3., 1.], [4.,5., 1.]], device)?;
    let B_inv = B.t()?;
    println!("tensor inverse: source={}; target={}",B, B_inv);
    // 判断矩阵是否相等
    println!("B == B.T={}", (B.eq(&B_inv)?));

    /// 可以根据不同的需要指定shape
    let X = Tensor::arange(0f32, 24f32, device)?
            .reshape((2, 3, 4))?;
    println!("===Tensor(2,3,4)={}", X);

    ///
    let A = Tensor::arange(0f32, 20f32, device)?
            .reshape((5, 4))?;
    let B = A.clone();
    println!("A={}, A+B={}, A*B={}", A, (&A+&B)?, (&A*&B)?);

    /// 矩阵的乘法与加法
    /// 矩阵加或乘一个常量并不会改变其shape
    let a = 2f64;
    let X = Tensor::arange(0f32, 24f32, device)?
            .reshape((2, 3, 4))?;
    println!("a+X={}, a*X={}, (a*X).shape={:?}", (a+&X)?, (a*&X)?, (a*&X)?.shape());

    // 注意数据类型保持一致
    let tensor_a = Tensor::new(&[a],device)?.to_dtype(DType::F32)?;
    println!("Method2: a+X={}, a*X={}, (a*X).shape={:?}",
             &X.broadcast_add(&tensor_a.reshape((1, 1))?)?,
             &X.broadcast_mul(&tensor_a)?,
             (a*&X)?.shape()
    );

    /// 降维
    let x = Tensor::arange(0f32, 4f32, device)?;
    println!("x={}, sum(x)={}, x.shape={:?}", x, x.sum(0)?,x.shape());
    /// 张量的求和
    /// sum((0,1)) 从轴0和1进行降维，即求和后得到一个标量;等同于sum_all()
    println!("A shape={:?}, Sum(A)={:?}， Sum(A(0))={:?}, Sum(A(1))={:?}, Sum_All(A)={:?}",
             A.shape()
             , A.sum((0,1))?
             , A.sum(0)?,
             A.sum(1)?,
            A.sum_all(),
    );
    /// 平均数
    println!("A_mean={:?}, A_sum/A_elemcount={:?}",
             A.mean((0,1))?
             , (A.sum_all()?.to_scalar::<f32>()?) / (A.elem_count() as f32));

    println!("A_mean_dim0={:?}, A_sum_dim0/A_dim0_elemcount={:?}"
             , A.mean(0)?
             ,(A.sum(0)? / (A.shape().dim(0)? as f64))? // 注意这里的除数类型必须是f64
    );

    /// 若是不进行降维求和
    println!("sum_keepdim0={}; sum_keepdim1={}"
             , A.sum_keepdim(0)?
             , A.sum_keepdim(1)?);

    /// 通过广播机制实现
    println!("A/A.sum_keepdim(1)={}", &A.broadcast_div(&A.sum_keepdim(1)?)? );

    /// cumsum是按照指定dim将元素进行累加
    /// 比如当dim=0时，是按照行进行的，第一行元素是不变的，接着第二行原有的元素加上第一行对应位置的元素，依次类推，得到与原来shape一样的累加结果
    /// 当dim=1时，是按照列进行的，每一行的元素第一列的元素不变，接着后面的元素加上其前面的元素值得到新的值并放置该位置，依次类推，得到与原来shape一样的累加结果
    println!("A_cumsum_dim0={}; A_cumsum_dim1={}", &A.cumsum(0)?, &A.cumsum(1)?);

    /// 向量的点积
    let y = Tensor::ones(4, DType::F32,device)?;
    /// 向量点积
    println!("x={}, y={}, dot(x,y)={}", x, y, Tensor::sum(&(&x * &y)?, 0)?);
    println!("broadcast_matmul={}", &x.broadcast_mul(&y)?);

    /// 矩阵和向量积
    println!("A={:?}, x={:?}, {}", A.shape(), x.shape(),
            &A.broadcast_mul(&x)?.sum(1)?
    );

    /// 矩阵和矩阵积
    let B = Tensor::ones((4,3), DType::F32, device)?;
    println!("A shape={:?}, B shape={:?}, A*B={}", A.shape(), B.shape(), &A.matmul(&B)?);

    Ok(())
}

/// 关于L1和L2范数是需要自己实现
/// 其他的范数可以参考：https://en.wikipedia.org/wiki/Norm_(mathematics)
/// TODO: Tensor中并没有对应的现成算法

/// 范数计算中的幂指数值
#[derive(Debug)]
enum P {
    L1,
    L2,
    L3, // 仅仅为了测试
}
// 2维矩阵L2范数
pub fn norm_forbenius(t: &Tensor, dims: &[usize], p: P) -> Result<Tensor> {
    match p {
        /// L1 范数
        /// X.abs 接着求和
        P::L1 => t.abs()?.sum(dims),
        /// 矩阵L2范数
        /// 先X^2，再求和，接着开平方
        P::L2 => t.sqr()?.sum(dims)?.sqrt(),
        _ => panic!("{:?}, {}", p," unimplemented norm"),
    }
}

pub fn norm_forbenius_scalar(t: &Tensor, dims: &[usize], p: P) -> Result<f32> {
    norm_forbenius(t, dims, p)?.to_scalar::<f32>()
}

fn candle_datas_process_normalize_demo(device: &Device) -> Result<()> {
    let u = Tensor::new(&[3.0f32, -4.0], device)?;
    //println!("u shape={:?}", u.shape());
    println!("L2: u shape={:?} L2_norm={:?}", u.shape(), norm_forbenius_scalar(&u, &[0], P::L2)?);
    println!("L1: u shape={:?}, L1_norm={:?}", u.shape(), norm_forbenius_scalar(&u, &[0], P::L1)?);

    let ones = Tensor::ones((4,9), DType::F32, device)?;
    println!("ones shape={:?}, L2_norm={:?}", ones.shape(), norm_forbenius_scalar(&ones, &[0,1], P::L2)?);
    // 为实现具体的norm，会panic
    //println!("ones shape={:?}, L2_norm={:?}", ones.shape(), norm_forbenius_scalar(&ones, &[0,1], P::L3)?);

    Ok(())
}

/// 微积分
fn candle_datas_process_diff_cal_demo(device: &Device) -> Result<()> {
    /// 模拟对函数 3 * x^2 - 4 * x求导
    fn f(x: f32) -> f32 {
        3.0 * x.powf(2f32) - 4.0 * x
    }

    fn numeric_lim(f: impl Fn(f32) -> f32, x:f32, h:f32) -> f32 {
        (f(x + h) - f(x)) / h
    }

    let mut h = 0.1f32;
    let x = 1f32;
    for i in 0 .. 5 {
        println!("i={},h={:.5}, f(x)={}, f(x+h)={}, lim={:.5}", i, h, f(x), f(x + h), numeric_lim(f, x, h));
        h *= 0.1;
    }
    Ok(())
}

/// plotters 与 candle tensor整合
/// 在rust中达到与matpotlib绘图库同样效果，可使用plotters crate来完成
/// [plotters](https://github.com/plotters-rs/plotters)
fn candle_datas_process_plotters_demo(device: &Device) -> Result<()> {
    // 构建数据
    let datas = Tensor::arange_step(0f32, 3f32, 0.1f32, device)?;

    // 绘图
    // 创建svg图片对象
    let root = SVGBackend::new("plot.svg", (640, 480)).into_drawing_area();
    // 图片对象的背景颜色填充
    root.fill(&WHITE).unwrap();
    // 构建chat
    let mut chart = ChartBuilder::on(&root)
        .caption("f(x)", ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32 ..3.2f32, -5f32..15.5f32).unwrap();
    // 绘制网格线
    chart.configure_mesh().draw().unwrap();

    // 绘制函数曲线
    chart.draw_series(LineSeries::new(
            datas.to_vec1::<f32>()?.iter().map(|x| (*x, 3.0 * x.powf(2f32) - 4.0 * x)),
            &BLUE,
        )).unwrap()
        .label("f(x) = 3.0 * x^2 - 4.0 * x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    // 绘制切线线
    chart.draw_series(LineSeries::new(
        datas.to_vec1::<f32>()?.iter().map(|x| (*x, 2.0 * x - 3.0)),
        &RED,
    )).unwrap()
        .label("Tangent line(x=1) = 2.0 * x - 3.0")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // 配置边框
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();

    // 保存图片
    root.present().unwrap();

    Ok(())
}

/// 导数
/// 在candle中，求导的方式是使用Var类型，
/// - Var类型是Tensor的包装类，
/// - Var类型具有backward方法，该方法可以求出当前Var类型对应的Tensor的导数，
///
/// 如下示例输出：
/// 源tensor： a=[0., 1., 2., 3.] Tensor[[4], f32],
/// 向量点积： y=[28.]             Tensor[[], f32],
/// 梯度：y = 4 * x
/// grad_a=[ 0.,  4.,  8., 12.]  Tensor[[4], f32]
fn candle_datas_process_partial_der_demo(device: &Device) -> Result<()> {
    let var_a = Var::new(&[0f32, 1., 2., 3.], device)?;
    let a = var_a.as_tensor();
    let y = (2.0f64 * Tensor::sum(&(a * a)?, 0)?)?;

    let grad = y.backward()?;
    let grad_a = grad.get(&a).expect("no grad for a");
    println!("a={}, y={}, grad={:?}, grad_a={}", a, y, grad, grad_a);
    // y = 2xTx梯度：4x
    println!("a grad: {}", grad_a.eq(&(a * 4f64)?)? );

    let y = a.sum(0)?;
    let grad = y.backward()?;
    // y = sum(a)的梯度: 1
    let grad_a = grad.get(&a).expect("no grad for a");
    println!("(y=sum(a)) grad: {:?}", grad_a );

    let y = (a * a)?;
    let grad = y.sum(0)?.backward()?;
    // y = a^2的梯度: 2a
    let grad_a = grad.get(&a).expect("no grad for a");
    println!("(y=a^2) grad: {:?}", grad_a );

    let y = (a * a)?;
    // 如下操作会分离出与y具有相同的值，梯度不再向后流经u到a了
    // 反向传播函数计算z=u * a关于a的偏导数,并将u用作常量处理
    let u = y.detach();
    let z = (&u * a)?;
    let grad_z = z.sum(0)?.backward()?;
    // z = u * a的梯度: u
    let grad_a = grad_z.get(&a).expect("no grad for a");
    println!("z=u * a; a.grad == u:{}", grad_a.eq(&u)?);

    // 在y上调用反向传播，得到y=a*a关于a的导数即2a
    let grad_y = y.sum(0)?.backward()?;
    let grad_a = grad_y.get(&a).expect("no grad for a");
    println!("y=a*a; a.grad == 2a:{}", grad_a.eq(&(a * 2f64)?)?);
    
    Ok(())
}

/// 控制流计算变量的梯度
/// 示例代码如下：
/// def f(a):
///     b = a * 2
///     while b.norm() < 1000:
///         b = b * 2
///     if b.sum() > 0:
///         c = b
///     else:
///         c = 100 * b
///     return c
/// 测试
/// a = torch.randn(size=(), requires_grad=True)
/// d = f(a)
/// d.backward()
/// a.grad == d / a
/// 输出：tensor(True)
///
/// 关于梯度grad这块会有专门的test demos [关于梯度](basic_grad_demos.rs)
fn candle_datas_process_partial_der_process_demo(device: &Device) -> Result<()> {
    fn f(x: &Tensor, device: &Device) -> Result<Var> {
        let con = Var::new(&[2f32], device)?;
        let xx = Var::from_tensor(x)?;
        let con_x = &xx.as_tensor().broadcast_mul(&con)?;
        let mut b = xx;
        b.set(&con_x)?;

        while norm_forbenius_scalar(b.as_tensor(), &[0,1], P::L2)? < 1000.0 {
            b.set(&(b.as_tensor().broadcast_mul(&con)?))?;
        }

        println!("sum_scalar={:?}", b.sum_all()?.to_scalar::<f32>()?);
        if b.sum_all()?.to_scalar::<f32>()?.lt(&0f32) {
            b.set(&(b.as_tensor().broadcast_mul(&Tensor::new(&[100f32], device)?)?))?;
        }

        Ok(b)
    }

    let x = Tensor::randn(0f32, 1f32, (3,4), device)?;
    let xx = Var::from_tensor(&x)?;
    let y = f(&xx, device)?;
    let grad = y.as_tensor().backward()?;
    let grad_x = grad.get(&xx).expect("no grad for x");
    println!("y={}, grad_x={}", y, grad_x);
    println!("y/xx == grad_x:{}",(grad_x).eq(&y.div(&xx)?)?);


    Ok(())
}

/// 概率计算
/// 在candle中，抽样是通过LogitsProcessor实现的
/// 抽样方式提供了：
///  ArgMax，All(可实现Multinomial抽样)，TopK，TopP，TopKThenTopP
/// 在定义抽样器时，有两种方式
/// - new(seed: u64, temperature: Option<f64>, top_p: Option<f64>)： 定义比较简单，但可定制比较固定
/// - from_sampling(seed: u64, sampling: Sampling) 支持按需指定抽样sampling方式
fn candle_datas_process_probability_demo(device: &Device) -> Result<()> {
    let fair_probs = (Tensor::ones(&[6], DType::F32, device)? / 6f64)?;
    println!("fair_probs={}", fair_probs);
    // 当将使用new时，并将top_p设置为None时，则可实现Multinomial抽样
    let mut logits_process = LogitsProcessor::new(1337, Some(1.7), None);
    for _ in 0..10 {
        let token = logits_process.sample(&fair_probs)?;
        println!("Multinomial: token={:?}", token);
    }

    logits_process = LogitsProcessor::from_sampling(1337, Sampling::All {
        temperature: 1.7,
    });
    for _ in 0..10 {
        let token = logits_process.sample(&fair_probs)?;
        println!("Multinomial from_sampling: token={:?}", token);
    }
    
    logits_process = LogitsProcessor::from_sampling(1337, Sampling::TopKThenTopP {
        k: 3,
        p: 0.5,
        temperature: 2.0,
    });
    for _ in 0..10 {
        let token = logits_process.sample(&fair_probs)?;
        println!("TopKThenTopP: token={:?}", token);
    }

    Ok(())
}

fn candle_datas_process_probability_demo_v2(device: &Device) -> Result<()> {
    let fair_probs = (Tensor::ones(&[6], DType::F32, device)? / 6f64)?;
    println!("fair_probs={}", fair_probs);
    // 当将使用new时，并将top_p设置为None时，则可实现Multinomial抽样
    let mut logits_process = LogitsProcessor::new(1337, Some(1.7), None);
    let mut ids = vec![11111;500];
    let mut datas = vec![-999.0;500];
    for _ in 0..500 {
        let token = logits_process.sample(&fair_probs)?;
        ids.push(token);
        // println!("index={:?} val={}", token, fair_probs.i(token as usize)?)
        datas.push(fair_probs.i(token as usize)?.to_scalar::<f32>()?);
    }
    let ids = ids.iter()
        .filter(|x| (*x).ne(&11111))
        .cloned()
        .collect::<Vec<u32>>()
        ;
    let datas = datas.iter()
        .filter(|x| (*x).ne(&-999.0f32))
        .cloned()
        .collect::<Vec<f32>>();
    let datas = Tensor::from_vec(datas, (20,25),device)?;
    let cum_counts = datas.cumsum(0)?;
    let estimates = &cum_counts.broadcast_div(&cum_counts.sum_keepdim(1)?)?;
    println!("cum_counts={}, estimates={:.8}", cum_counts, estimates);

    //

    Ok(())
}

fn main() {
    let device = Device::Cpu;
    candle_datas_process_probability_demo_v2(&device);
    // candle_datas_process_probability_demo(&device);
    //candle_datas_process_partial_der_process_demo(&device);
    // candle_datas_process_partial_der_demo(&device);
    //candle_datas_process_plotters_demo(&device);
    //candle_datas_process_diff_cal_demo(&device);
    // candle_datas_process_normalize_demo(&device);
    //candle_liner_algebra_basic_demo(&device);
    // candle_datas_process_nan_demo(&device);
    // 操作数据集
    // read_data_source_demo(&device);
    // Tensor基本操作
    // candle_datas_process_demo(&device);
}
