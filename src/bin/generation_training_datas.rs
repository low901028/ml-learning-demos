#![feature(float_gamma)]

use std::cmp::max;
use std::ops::{Div, DivAssign};
use candle_core::{DType, Device, IndexOp, Result, Tensor, WithDType, D};
use ndarray::{range, s, Array, Array1, Array2, ArrayBase, Axis, Zip};
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::{random, thread_rng};
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::real::Real;

fn vector_to_tensor_demo<T: WithDType>(datas: Vec<T>) -> Result<Tensor> {
    Tensor::try_from(datas)
}

/// 无论是slice_assign还是gather两个方法的操作
/// 相当于在对应的shape中获取其需要的一块“子shape”
/// 要确保目标shape隶属于源shape的“子集”
/// 比如slice_assign：
///    true_w.slice_assign(&[0..4, 0..1], &true_w_four) 源shape=(20,1),目标shape=(4，1)
///    这里面的源shape和目标shape的列维度都是一致的，若是目标shape=(2,1)，也是满足满足目标shape是源shape的“子集”
///    换句话就是行维度和列维度不能超过源shape的即可
///    相当于说获取true_w中的4行1列数据，
/// 再比如gather：
///    poly_features.gather(&Tensor::arange(0u32, 40, device)?.reshape((2,20))? 源shape=(200,20), 目标shape=(2,20)
///    这里源和目标shape的列维度也是一致的，若是目标shape=(2,10)也是可以的还是满足目标shape是源shape的“子集”；
///    行维度和列维度不能超过源shape的即可
///  相当于获取poly_features的2行20列
///  
fn generate_training_dataset_demo(device: &Device) -> Result<()> {
    let max_degree = 20;
    let (n_train, n_test) = (100, 100);
    let total_samples = n_train + n_test;

    let true_w = Tensor::zeros((max_degree, 1 ), DType::F32, device)?;
    let true_w_four = Tensor::new(&[5f32, 1.2, -3.4, 5.6], device)?.reshape((4,1))?;
    let true_w = &true_w.slice_assign(&[0..4, 0..1], &true_w_four)?.flatten_from(0)?;
    println!("true_w={}", true_w);

    let features = Tensor::randn(0f32,1f32, (total_samples, 1), device)?;
    println!("features shape={:?}", features.shape());
    let arange_data = Tensor::arange(0f32, max_degree as f32, device)?;
    let arange_datas = arange_data.reshape(((),1))?;
    println!("arange_datas={:?}", arange_datas.shape());
    let poly_features = features.broadcast_pow(&arange_datas.t()?)?;
    
    println!("poly features={}", poly_features);
    
    let len = poly_features.dim(0)?;

    let gammas = (0..max_degree).map(|i|{
        let gamma = f32::gamma((i+1) as f32);
        gamma
    }).collect::<Vec<f32>>();
    
    let gamma_tensor = Tensor::from_vec(gammas.clone(), max_degree, device)?;
    let replace = Tensor::ones(poly_features.shape(), DType::F32, device)?.broadcast_mul(&gamma_tensor)?;
    let  poly_features = poly_features.slice_assign(&[0..len, 0..max_degree], &replace)?;
    println!("after modify poly_features={} shape={:?}", poly_features, poly_features.shape());
    
    let labels = poly_features.broadcast_mul(&true_w)?;
    let labels = Tensor::randn(0f32, 0.1, labels.shape(), device)?.add(&labels)?;
    println!("after modify labels shape={:?}", labels.shape());

    println!("features[:2]={}, poly_features[:2, :]={}, labels[:2]={}"
        ,features.gather(&Tensor::arange(0u32, 2, device)?.reshape((2,1))?, 0)?
        ,poly_features.gather(&Tensor::arange(0u32, 40, device)?.reshape((2,20))?, 0)?
        ,labels.gather(&Tensor::arange(0u32, 2, device)?.reshape((1,2))?, 0)?
    );
    
    Ok(())
}

fn main() {
    let device = Device::Cpu;
    generate_training_dataset_demo(&device).unwrap();
}