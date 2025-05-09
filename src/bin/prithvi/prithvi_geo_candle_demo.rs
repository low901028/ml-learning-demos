mod tiff_file_demo;
mod geotiff_candle_demo;
mod vit_privith_main;
mod vit_main;
mod privith_geotiff_3d_candle_demo;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use clap::Parser;

use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;
use tiff::ColorType;

use std::fs::File;
use std::path::PathBuf;
use candle_core::op::Op;
use candle_nn::VarBuilder;
use candle_transformers::models::vit;
use serde::Deserialize;
use tiff::encoder::TiffValue;

#[derive(Parser, Debug, Clone)]
pub struct Args {
    /// 测试文件夹，其中文件格式：tif.
    #[arg(long,default_value="prithvi-model/tifs/alaska")]
    data_files: String,

    /// 本地模型路径
    #[arg(long)]
    model: Option<String>,

    /// 索引集，可选项
    #[arg(long)]
    input_indics: Option<Vec<String>>,
}

/// ==========================
/// 解析geo tiff 文件
/// ==========================
fn parse_tiff(filename: &std::path::Path) -> candle_core::Result<Tensor> {
    let mut decoder = Decoder::new(File::open(filename)?).unwrap();
    decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());
    let _dimensions = decoder.dimensions().unwrap() else {
        panic!("Cannot get dimensions!!!")
    };
    let (w, h) = (_dimensions.0 as usize, _dimensions.1 as usize);
    let color_type = decoder.colortype().unwrap() else {
        panic!("Cannot get colortype!!!")
    };
    let DecodingResult::I16(datas) = decoder.read_image().unwrap() else {
        panic!("Cannot read band data")
    };
    let elem_size = datas.len();
    println!("{:?}", datas.len());
    let datas = datas.iter().map(|d| d.to_owned() as f32).collect::<Vec<f32>>();
    let tensor = Tensor::from_vec(datas, (elem_size/w/h, w, h), &Device::Cpu)?;

    Ok(tensor)
}

const  BATCH_SIZE:i32 = 16i32;
const EPOCHS: i32 = 50i32;
const LR: f64 = 2.0e-4;
/// 衰变率
const WEIGHT_DECAY: f64 = 0.1;

const HEAD_DROPOUT: f64 =0.1;
const FREEZE_BACKBONE: bool = false;

const BANDS: [&'static str;6] = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"];
const NUM_FRAMES: i32 = 3;
/// 分类权重
const CLASS_WEIGHTS: [f64;13] = [
    0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462,
    1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702
];

const MODEL_PATH: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/Prithvi_EO_V2_300M_TL.pt";
/// ==============================
/// 处理geo tiff对应的Tensor
/// ==============================
fn geo_tiff_process(tensor: &Tensor, model: &str, device: &Device) -> candle_core::Result<()>{
    let model_path = std::path::PathBuf::from(model);
    if model_path.exists() {
        println!("===model exists, load model from {}", model_path.display());
        // 加载模型
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? };
        let vb_m = vb.pp("model");
    } else {
        println!("===model not exists, train model from {}", model_path.display());
        // 训练模型
    }

    Ok(())
}

/// ==============================
///  处理文件夹中的tif文件
/// ==============================
fn load_dir_tifs<T: AsRef<std::path::Path>>(dir: T, device: &Device) -> candle_core::Result<()> {
    let tifs = require_dir_tifs(dir)?;
    for tif in tifs {
        let tif_path = std::path::PathBuf::from(tif);
        let _tensor = parse_tiff(&tif_path)?;
        let _geo_tif = geo_tiff_process(&_tensor, MODEL_PATH, device);
    }
    Ok(())
}

pub fn require_dir_tifs<T: AsRef<std::path::Path>>(dir: T) -> candle_core::Result<Vec<String>> {
    let dir = dir.as_ref();
    let mut tifs = vec![];
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if path.extension().unwrap() == "tif" {
                tifs.push(path.display().to_string())
            }
        }
    }
    Ok(tifs)
}

fn geo_tiff_demo(device: &Device) -> candle_core::Result<()> {
    let args = Args::parse();
    let tifs = load_dir_tifs(args.data_files, &device)?;

    Ok(())
}

/// ==============================
/// 基于candle-nn的vit模式测试
/// ==============================
fn load_image_with_std_mean<P: AsRef<std::path::Path>>(
    p: P,
    res: usize,
    mean: &[f32; 6],
    std: &[f32; 6],
    device: &Device
) -> candle_core::Result<Tensor> {
    let filename = p.as_ref();
    let mut decoder = Decoder::new(File::open(filename)?).unwrap();
    decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());
    let _dimensions = decoder.dimensions().unwrap() else {
        panic!("Cannot get dimensions!!!")
    };
    let (w, h) = (_dimensions.0 as usize, _dimensions.1 as usize);
    let color_type = decoder.colortype().unwrap() else {
        panic!("Cannot get colortype!!!")
    };
    let DecodingResult::I16(datas) = decoder.read_image().unwrap() else {
        panic!("Cannot read band data")
    };

    let raw_data = datas.data().to_vec();
    // let img = image::ImageReader::new(std::io::Cursor::new(raw_data))
    //     .decode()
    //     .map_err(candle_core::Error::wrap)?
    //     .resize_to_fill(
    //         res as u32,
    //         res as u32,
    //         image::imageops::FilterType::Triangle,
    //     );
    // let img = img.to_rgb8();
    // let data = img.into_raw();
    let data = raw_data;
    let data = Tensor::from_vec(data, (w, h, 6, 2), device)?.permute((3, 2, 0, 1))?;
    let mean = Tensor::new(mean, device)?.reshape((6, 1, 1))?;
    let std = Tensor::new(std, device)?.reshape((6, 1, 1))?;

    (data.to_dtype(candle_core::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

pub const IMAGENET_MEAN: [f32; 6] = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0];
pub const IMAGENET_STD: [f32; 6] = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0];
fn load_image224(path: String, device: &Device) -> candle_core::Result<Tensor> {
    load_image_with_std_mean(path, 224, &IMAGENET_MEAN, &IMAGENET_STD, device)
}

fn candle_geo_demo(device: &Device) -> candle_core::Result<()> {
    let args = Args::parse();
    let device = candle_core::Device::Cpu;
    let tifs = require_dir_tifs(args.data_files)?;
    for tif in tifs {
        let image = load_image224(tif, &device)?.to_device(&device)?;
        println!("loaded image {image:?}");

        let model_file = match &args.model {
            None => {
                // let api = hf_hub::api::sync::Api::new().unwrap();
                // let api = api.model("google/vit-base-patch16-224".into());
                // api.get("model.safetensors").unwrap()
                "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/Prithvi_EO_V2_300M_TL.safetensors".to_string()
            }
            Some(model) => model.into(),
        };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)?
        };

        let config_file = std::path::PathBuf::from(r"/Users/dalan/rustspace/ml-learning-demos/prithvi-model/config.json");
        let raw_data = std::fs::read(&config_file)?;
        // let config: PrithviConfig = serde_json::from_slice(&raw_data).unwrap();
        // println!("config ={:?}", config);
    }

    Ok(())
}

fn main() -> candle_core::Result<()>{
    println!("===prithvi gemo....");
    let device = candle_core::Device::Cpu;
    // geo_tiff_demo(&device);
    candle_geo_demo(&device).unwrap();

    Ok(())
}