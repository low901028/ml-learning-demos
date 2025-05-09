use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use tiff::{ColorType, decoder::Decoder};
use std::fs::File;
use std::path::Path;
use candle_nn::{linear, Linear, VarBuilder, VarMap};
use tiff::decoder::DecodingResult;
use tiff::encoder::TiffValue;

// 模型配置结构
#[derive(Debug, Clone,PartialEq, serde::Deserialize, serde::Serialize, Default)]
pub struct PrithviPretrainConfig {
    pub img_size: usize,
    pub num_frames: usize,
    pub patch_size: [usize;3],
    pub in_chans: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub decoder_embed_dim: usize,
    pub decoder_depth: usize,
    pub decoder_num_heads: usize,
    pub mlp_ratio: f32,
    pub coords_encoding: [String;2],
    pub coords_scale_learn: bool,
    pub mask_ratio: f32,
    pub norm_pix_loss: bool,
    pub bands : [String ;6],
    pub mean : [f32;6],
    pub std : [f32;6],
}
#[derive(Debug, Clone,PartialEq, serde::Deserialize, serde::Serialize, Default)]
pub struct ModelConfig {
    pub architecture: String,
    pub num_features: usize,
    pub pretrained_cfg: PrithviPretrainConfig,
}

impl ModelConfig {
    fn from_json(config: &str) -> Self {
        let config: ModelConfig = {
            let config_file = std::path::PathBuf::from(config);
            serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap()
        };
        config
    }
}

// 预处理模块
struct GeoTiffProcessor {
    config: ModelConfig,
    device: Device,
}

impl GeoTiffProcessor {
    fn new(config: ModelConfig, device: Device) -> Self {
        Self { config, device }
    }

    fn process<P: AsRef<std::path::Path>>(&self, config: &ModelConfig, path: &P) -> Result<Tensor> {
        // 读取TIFF文件
        let filename = path.as_ref();
        let mut decoder = tiff::decoder::Decoder::new(File::open(filename)?).unwrap();
        decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());
        let _dimensions = decoder.dimensions().unwrap() else {
            panic!("Cannot get dimensions!!!")
        };
        let (w, h) = (_dimensions.0 as usize, _dimensions.1 as usize);
        let color_type = decoder.colortype().unwrap() else {
            panic!("Cannot get colortype!!!")
        };
        println!("color type={:?}", color_type);

        let DecodingResult::I16(datas) = decoder.read_image().unwrap() else {
            panic!("Cannot read band data")
        };

        let img = datas.data().to_vec();
        // 转换为Tensor [C, H, W]
        let img_tensor = self.image_to_tensor(config, img, color_type)?;

        // 中心裁剪到224x224
        let cropped = self.center_crop(img_tensor)?;

        // 归一化处理
        self.normalize(cropped)
    }

    fn image_to_tensor(&self, config: &ModelConfig, img: Vec<u8>, color_type: ColorType) -> Result<Tensor> {
        let (h, w) = (448, 560);
        let c = match color_type {
            ColorType::RGB(..) => 3,
            ColorType::RGBA(..) => 4,
            ColorType::Multiband {..} => 12,
            _ => panic!("Unsupported color type"),
        };

        Tensor::from_vec(img, (h, w, c), &self.device)?
            .permute((2, 0, 1))? // [C, H, W]
            .to_dtype(DType::F32)
    }

    fn center_crop(&self, img: Tensor) -> Result<Tensor> {
        let (_, h, w) = img.dims3()?;
        let top = (h - self.config.pretrained_cfg.img_size) / 2;
        let left = (w - self.config.pretrained_cfg.img_size) / 2;
        img.narrow(1, top, self.config.pretrained_cfg.img_size)?
            .narrow(2, left, self.config.pretrained_cfg.img_size)
    }

    fn normalize(&self, img: Tensor) -> Result<Tensor> {
        let mean = Tensor::from_vec(
            self.config.pretrained_cfg.mean.clone().to_vec(),
            (self.config.pretrained_cfg.in_chans, 1, 1),
            &self.device
        )?;

        let std = Tensor::from_vec(
            self.config.pretrained_cfg.std.clone().to_vec(),
            (self.config.pretrained_cfg.in_chans, 1, 1),
            &self.device
        )?;

        (img - mean)?.broadcast_div(&std)
    }
}

// ViT模型实现
struct ViT {
    patch_embed: candle_nn::Conv2d,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: candle_nn::LayerNorm,
    head: candle_nn::Linear,
}

impl ViT {
    fn new(cfg: &ModelConfig, vb_params: VarBuilder, device: &Device) -> Result<Self> {
        let patch_embed = candle_nn::conv2d(
            cfg.pretrained_cfg.in_chans,
            cfg.pretrained_cfg.embed_dim,
            cfg.pretrained_cfg.patch_size[1],
            candle_nn::Conv2dConfig {
                stride: cfg.pretrained_cfg.patch_size[2],
                ..Default::default()
            },
            vb_params.clone(),
        )?;

        let num_patches = (cfg.pretrained_cfg.img_size / cfg.pretrained_cfg.patch_size[1]) * (cfg.pretrained_cfg.img_size / cfg.pretrained_cfg.patch_size[2]);
        let pos_embed = Tensor::randn(
            0.0,
            1.0,
            (1, num_patches, cfg.pretrained_cfg.embed_dim),
            device,
        )?;

        let mut blocks = Vec::with_capacity(cfg.pretrained_cfg.depth);
        for _ in 0..cfg.pretrained_cfg.depth {
            blocks.push(Block::new(vb_params.clone(),cfg.pretrained_cfg.embed_dim, cfg.pretrained_cfg.num_heads, device)?);
        }

        let norm = candle_nn::layer_norm(cfg.pretrained_cfg.embed_dim, 1e-6, vb_params.clone())?;
        let head = candle_nn::linear(cfg.pretrained_cfg.embed_dim, cfg.num_features, vb_params.clone())?;

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            norm,
            head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, h, w) = x.dims3()?;
        let x = self.patch_embed.forward(x)?;
        let x = x.flatten_from(2)?.permute((0, 2, 1))?; // [B, N, C]
        let x = x.broadcast_add(&self.pos_embed)?;

        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        let x = self.norm.forward(&x)?;
        self.head.forward(&x.mean(D::Minus1)?)
    }
}

// Transformer Block
struct Block {
    attn: MultiHeadAttention,
    mlp: MLP,
}

impl Block {
    fn new(vb: VarBuilder,embed_dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let attn = MultiHeadAttention::new(vb.pp("attn"), embed_dim, num_heads)?;
        let mlp = MLP::new(embed_dim, 4 * embed_dim, vb)?;
        Ok(Self { attn, mlp })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.attn.forward(x)?;
        self.mlp.forward(&x)
    }
}

/// MultiHeadAttention
struct MultiHeadAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
}

impl MultiHeadAttention {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let qkv = linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;
        let scale = 1.0 / ((dim / num_heads) as f64).sqrt();

        Ok(Self { qkv, proj, num_heads, scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let qkv = self.qkv.forward(x)?.reshape((b, n, 3, self.num_heads, c / self.num_heads))?;

        let q = qkv.i((.., .., 0))?.contiguous()?;
        let k = qkv.i((.., .., 1))?.contiguous()?;
        let v = qkv.i((.., .., 2))?.contiguous()?;

        let attn = (q.matmul(&k.t()?)? * self.scale)?;
        let attn = candle_nn::ops::softmax(&attn, 4)?;

        let x = attn.matmul(&v)?.reshape((b, n, c))?;
        self.proj.forward(&x)
    }
}

// MLP模块
struct MLP {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    gelu: candle_nn::Activation,
}

impl MLP {
    fn new(in_dim: usize, hidden_dim: usize, vb_params: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(in_dim, hidden_dim, vb_params.clone())?;
        let fc2 = candle_nn::linear(hidden_dim, in_dim, vb_params.clone())?;
        Ok(Self {
            fc1,
            fc2,
            gelu: candle_nn::Activation::Gelu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.fc1)?
            .apply(&self.gelu)?
            .apply(&self.fc2)
    }
}

// 加载预训练权重
fn load_pretrained(path: &str, config: &ModelConfig,device: &Device) -> Result<ViT> {
    let mut varmap = VarMap::new();
    varmap.load(path)?;

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let model = ViT::new(
        config,
        vb,
        device,
    )?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use crate::require_dir_tifs;
    use super::*;

    pub const MODEL_PATH: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/Prithvi_EO_V2_300M_TL.safetensors";
    pub const DATASETS_PATH_TEST: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska";

    pub const DATASETS_PATH_TRAIN: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/florida";
    pub const CONFIG_PATH: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/config.json";

    #[test]
    fn test_preprocessing() -> Result<()> {
        let config = ModelConfig::from_json(CONFIG_PATH);
        let processor = GeoTiffProcessor::new(config.clone(), Device::Cpu);


        let tifs = require_dir_tifs(DATASETS_PATH_TRAIN)?;
        for tif in tifs {
            let file = std::path::PathBuf::from(&tif);
            let processed = processor.process(&config.clone(), &file)?;
            assert_eq!(processed.dims3()?, (6, 224, 224));
        }
        Ok(())
    }

    #[test]
    fn test_vit_forward() -> Result<()> {
        let config = ModelConfig::from_json(CONFIG_PATH);
        let device = Device::Cpu;

        let vit = load_pretrained(MODEL_PATH,&config.clone(), &Device::Cpu)?;
        let tifs = require_dir_tifs(DATASETS_PATH_TEST)?;
        let processor = GeoTiffProcessor::new(config.clone(), Device::Cpu);

        for tif in tifs {
            let file = std::path::PathBuf::from(&tif);
            let test_input = processor.process(&config.clone(), &file)?;
            let output = vit.forward(&test_input)?;
            assert_eq!(output.dims(), &[1024]);
        }
        Ok(())
    }
}