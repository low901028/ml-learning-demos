use candle_core::{D, DType, Device, Module, ModuleT, Result, Shape, Tensor, Var};
use candle_datasets::vision::Dataset;
use candle_nn::{Activation, Init, Linear, Optimizer, VarBuilder, VarMap};
use rand::seq::SliceRandom;
use std::collections::HashMap;

fn candle_activation_func_simple_demo(device: &Device) -> Result<()> {
    let x = Tensor::arange_step(-0.8, 8.0, 0.1, &device)?;
    let var = Var::from_tensor(&x)?;

    // /// ReLu
    let y = var.relu()?;
    // let yy = relu(&var)?;
    // println!("y={}, yy={}", y, yy);

    /*
    // grad
    let grad = y.backward()?;
    let grad_x = grad.get(&var.as_tensor()).expect("no grad");
    println!("grad_x={}", grad_x);
    // 对比relu前后数据的变化
    println!("x={}, y={}", x, y);

    /// sigmoid
    let y = candle_nn::ops::sigmoid(&var)?;
    let grad = y.backward()?;
    let grad_x = grad.get(&var.as_tensor()).expect("no grad");
    println!("sigmoid grad_x={}", grad_x);
    println!("x={}, y={}", x, y);

    /// tanh
    let y = var.tanh()?;
    let grad = y.backward()?;
    let grad_x = grad.get(&var.as_tensor()).expect("no grad");
    println!("tanh grad_x={}", grad_x);
    println!("x={}, y={}", x, y);
    */

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

fn multiply_layer_perceptron_train_demo(
    train_images: &Tensor,
    train_labels: &Tensor,
    test_images: &Tensor,
    test_labels: &Tensor,
    batch_size: usize,
    num_inputs: usize,
    params: &[Var; 4],
) -> Result<()> {
    ///
    /// 自定义Relu
    ///
    fn relu(x: &Tensor) -> Result<Tensor> {
        let a = Tensor::zeros_like(x)?;
        Tensor::maximum(x, &a)
    }

    fn net(x: &Tensor, num_inputs: usize, params: &[Var; 4]) -> Result<Tensor> {
        let x = x.clone().reshape(((), num_inputs))?;
        let h = relu(&x.matmul(&params[0])?.broadcast_add(&params[1])?)?;
        let y = h.matmul(&params[2])?.broadcast_add(&params[3]);

        y
    }

    fn loss(x: &Tensor, target: &Tensor) -> Result<Tensor> {
        candle_nn::loss::cross_entropy(x, target)
    }

    /// 数据样本
    let num_epoches = 10;
    let lr = 0.1;
    let n_batches = train_images.dim(0)? / batch_size;
    let mut batch_indexs = (0..n_batches).collect::<Vec<usize>>();
    let mut updater = candle_nn::optim::SGD::new(params.to_vec(), lr)?;

    for epoch in 0..num_epoches {
        println!("epoch={}", epoch);
        // 损失值总和
        let mut sum_loss = 0f32;
        batch_indexs.shuffle(&mut rand::thread_rng());
        for batch_idx in batch_indexs.iter() {
            let batch_train_images = train_images.narrow(0, batch_size * batch_idx, batch_size)?;
            let batch_train_labels = train_labels.narrow(0, batch_size * batch_idx, batch_size)?;

            let net = net(&batch_train_images, num_inputs, &params)?;
            let loss = loss(&net, &batch_train_labels)?;
            updater.backward_step(&loss)?;

            sum_loss += loss.to_vec0::<f32>()?;
        }
        // 计算平均损失值
        let avg_loss = sum_loss / n_batches as f32;
        // 训练测试集的图像集
        let test_logits = net(&test_images, num_inputs, &params)?;
        // 计算模型在测试集上正确分类的样本数量
        let sum_ok = test_logits
            // 对 test_logits 进行 argmax 操作，返回沿着最后一个维度的最大值的索引
            .argmax(D::Minus1)?
            // 将上一步得到的索引与 测试集的标签集 进行逐元素比较，生成一个布尔类型的张量，表示对应位置的元素是否相等
            .eq(&*test_labels)?
            // 将上一步得到的布尔类型的张量转换为 f32 类型的张量，其中 true 被转换为 1.0，false 被转换为 0.0
            .to_dtype(DType::F32)?
            // 对上一步得到的张量进行求和操作，将所有元素相加得到一个标量（scalar）
            .sum_all()?
            // 将上一步得到的标量转换为 f32 类型的数值
            .to_scalar::<f32>()?;
        // 正确分类的样本数量除以测试集中样本的总数，以得到模型在测试集上的准确率
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }

    Ok(())
}
fn multiply_layer_perceptron_demo(device: &Device) -> Result<()> {
    // candle_activation_func_simple_demo(device);

    let batch_size = 256;
    let dataset = load_fashion_mnist()?;

    // train数据集
    let train_images = dataset.train_images;
    let train_labels = dataset.train_labels.to_device(&device)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&device)?;

    // test数据集
    let test_images = dataset.test_images;
    let test_labels = dataset.test_labels;
    let test_labels = test_labels.to_dtype(DType::U32)?.to_device(&device)?;

    ///
    let (num_inputs, num_outputs, num_hiddens) = (784, 10, 256);
    let w1 = (Tensor::randn(0.0, 1.0, (num_inputs, num_hiddens), &device)?
        .to_dtype(DType::F32)?
        * 0.01)?;
    println!("w1={:?}", &w1.clone().shape());
    let var_w1 = Var::from_tensor(&w1)?;
    let b1 = Tensor::zeros((num_hiddens), DType::F32, &device)?;
    let var_b1 = Var::from_tensor(&b1)?;
    println!("b1={:?}", &b1.clone().shape());

    let w2 = (Tensor::randn(0.0, 1.0, (num_hiddens, num_outputs), &device)?
        .to_dtype(DType::F32)?
        * 0.01)?;
    println!("w2={:?}", &w2.clone().shape());
    let var_w2 = Var::from_tensor(&w2)?;
    let b2 = Tensor::zeros((num_outputs), DType::F32, &device)?;
    let var_b2 = Var::from_tensor(&b2)?;
    println!("b2={:?}", &b2.clone().shape());

    let params = [var_w1, var_b1, var_w2, var_b2];
    multiply_layer_perceptron_train_demo(
        &train_images,
        &train_labels,
        &test_images,
        &test_labels,
        batch_size,
        num_inputs,
        &params,
    )?;

    Ok(())
}

fn simple_multi_layer_perceptron_demo(device: &Device) -> Result<()> {
    struct Flatten;
    impl Module for Flatten {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            x.flatten_from(1)
        }
    }

    #[derive(Debug, Clone)]
    struct LinearModel {
        linear: Linear,
    }

    impl LinearModel {
        fn new(num_inputs: usize, num_outputs: usize, path: &str, vb: VarBuilder) -> Result<Self> {
            let w = vb.pp(path);
            let liner = Self {
                linear: candle_nn::linear(num_inputs, num_outputs, w)?,
            };

            Ok(liner)
        }
    }

    impl Module for LinearModel {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            self.linear.forward(x)
        }
    }

    struct Net {
        pub flatten: Flatten,
        linear1: LinearModel,
        relu: Activation,
        linear2: LinearModel,
    }

    impl Net {
        fn new(
            num_inputs: usize,
            num_hiddens: usize,
            num_outputs: usize,
            vb: VarBuilder,
        ) -> Result<Self> {
            let net = Net {
                flatten: Flatten,
                linear1: LinearModel::new(num_inputs, num_hiddens, "linear1", vb.clone())?,
                relu: Activation::Relu,
                linear2: LinearModel::new(num_hiddens, num_outputs, "linear2", vb.clone())?,
            };

            Ok(net)
        }
    }

    impl Module for Net {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let x = self.flatten.forward(x)?;
            let x = self.linear1.forward(&x)?;
            let x = self.relu.forward(&x)?;
            let x = self.linear2.forward(&x)?;
            Ok(x)
        }
    }

    let batch_size = 256;
    let dataset = load_fashion_mnist()?;

    // train数据集
    let train_images = dataset.train_images;
    let train_labels = dataset.train_labels.to_device(&device)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&device)?;

    // test数据集
    let test_images = dataset.test_images;
    let test_labels = dataset.test_labels;
    let test_labels = test_labels.to_dtype(DType::U32)?.to_device(&device)?;

    ///
    let (num_inputs, num_outputs, num_hiddens) = (784, 10, 256);

    let weights_1 =
        Tensor::randn(0.0, 0.01, (num_inputs, num_hiddens), &device)?.to_dtype(DType::F32)?;
    let var_weights_1 = Var::from_tensor(&weights_1)?;

    let weights_2 =
        Tensor::randn(0.0, 0.01, (num_hiddens, num_outputs), &device)?.to_dtype(DType::F32)?;
    let var_weights_2 = Var::from_tensor(&weights_2)?;

    let mut weights = HashMap::new();
    weights.insert("linear1.weight".to_string(), weights_1.clone().t()?);
    weights.insert("linear2.weight".to_string(), weights_2.clone().t()?);

    // let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let mut var_map = VarMap::new();
    var_map.get((num_hiddens,num_inputs), "linear1.weight", Init::Randn { mean: 0.0, stdev: 0.01}, DType::F32, &Device::Cpu)?;
    var_map.get((num_outputs,num_hiddens), "linear2.weight", Init::Randn { mean: 0.0, stdev: 0.01}, DType::F32, &Device::Cpu)?;
    var_map.get(num_hiddens, "linear1.bias", Init::Const(0.0), DType::F32, &Device::Cpu)?;
    var_map.get(num_outputs, "linear2.bias", Init::Const(0.0), DType::F32, &Device::Cpu)?;

    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = Net::new(num_inputs, num_hiddens, num_outputs, vb)?;

    /// 数据样本
    let num_epoches = 10;
    let lr = 0.1;
    let n_batches = train_images.dim(0)? / batch_size;
    let mut batch_indexs = (0..n_batches).collect::<Vec<usize>>();
    // let mut updater = candle_nn::optim::SGD::new(
    //     vec![
    //         Var::from_tensor(&weights_1.clone())?,
    //         Var::from_tensor(&weights_2.clone())?,
    //     ],
    //     lr,
    // )?;

    let mut updater = candle_nn::optim::SGD::new(
        var_map.all_vars(),
        lr,
    )?;

    for epoch in 0..num_epoches {
        println!("epoch={}", epoch);
        // 损失值总和
        let mut sum_loss = 0f32;
        batch_indexs.shuffle(&mut rand::thread_rng());
        for batch_idx in batch_indexs.iter() {
            let batch_train_images = train_images.narrow(0, batch_size * batch_idx, batch_size)?;
            let batch_train_labels = train_labels.narrow(0, batch_size * batch_idx, batch_size)?;

            let net = net.forward(&batch_train_images)?;
            let loss = candle_nn::loss::cross_entropy(&net, &batch_train_labels)?;
            updater.backward_step(&loss)?;

            sum_loss += loss.to_vec0::<f32>()?;
        }
        // 计算平均损失值
        let avg_loss = sum_loss / n_batches as f32;
        // 训练测试集的图像集
        let test_logits = net.forward(&test_images)?;
        // 计算模型在测试集上正确分类的样本数量
        let sum_ok = test_logits
            // 对 test_logits 进行 argmax 操作，返回沿着最后一个维度的最大值的索引
            .argmax(D::Minus1)?
            // 将上一步得到的索引与 测试集的标签集 进行逐元素比较，生成一个布尔类型的张量，表示对应位置的元素是否相等
            .eq(&test_labels)?
            // 将上一步得到的布尔类型的张量转换为 f32 类型的张量，其中 true 被转换为 1.0，false 被转换为 0.0
            .to_dtype(DType::F32)?
            // 对上一步得到的张量进行求和操作，将所有元素相加得到一个标量（scalar）
            .sum_all()?
            // 将上一步得到的标量转换为 f32 类型的数值
            .to_scalar::<f32>()?;
        // 正确分类的样本数量除以测试集中样本的总数，以得到模型在测试集上的准确率
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }

    Ok(())
}

fn main() {
    let device = Device::Cpu;
    simple_multi_layer_perceptron_demo(&device).unwrap();
    // multiply_layer_perceptron_demo(&device);
}
