// Cargo.toml新增依赖
/*
[dependencies]
candle-safetensors = "0.8.4"  # 新增safetensors支持
*/
use std::path::Path;

// 修改后的src/main.rs
pub(crate) mod prithvi {
    use std::fs::File;
    use candle_core::{Result, DType, Device, Tensor, Module, IndexOp};
    use candle_nn::{linear, Linear, VarBuilder, VarMap};
    // use candle_safetensors::Load;  // 新增safetensors支持
    use tiff::decoder::{Decoder, DecodingResult};
    use std::path::Path;
    use candle_transformers::models::dac::DecoderBlock;
    use tiff::ColorType;

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
    pub struct PrithviConfig {
        pub architecture: String,
        pub num_features: usize,
        pub pretrained_cfg: PrithviPretrainConfig,
    }

    impl PrithviConfig {
        fn from_json(config: &str) -> Self {
            let config: PrithviConfig = {
                let config_file = std::path::PathBuf::from(config);
                serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap()
            };
            config
        }
    }
    // 更新后的GeoTIFF加载器
    pub struct GeoTiffLoader {
        device: Device,
        mean: Tensor,
        std: Tensor,
        config: PrithviConfig,
    }

    impl GeoTiffLoader {
        pub fn new(device: Device, config: &PrithviConfig) -> Self {
            // 从配置中加载归一化参数
            let mean = Tensor::new(
                &[1087.0f32, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
                &device
            ).unwrap().unsqueeze(0).unwrap();

            let std = Tensor::new(
                &[2248.0f32, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
                &device
            ).unwrap().unsqueeze(0).unwrap();

            let mean = Tensor::new(
                &config.pretrained_cfg.mean,
                &device
            ).unwrap().unsqueeze(0).unwrap();

            let std = Tensor::new(
                &config.pretrained_cfg.std,
                &device
            ).unwrap().unsqueeze(0).unwrap();

            let config = config.clone();
            Self { device, mean, std, config }
        }

        pub fn load<P: AsRef<std::path::Path>>(&self, path: P) -> Result<Tensor> {
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

            let data: Vec<f32> = datas.chunks_exact(6)  // 处理6通道数据
                .flat_map(|chunk| {
                    chunk.iter().map(|&v| v as f32)
                })
                .collect();

            let tensor = Tensor::from_vec(data, (6, h, w), &self.device)?
                .permute((2, 0, 1))?
                .to_dtype(DType::F32)?;

            // 图片裁剪
            let tensor = self.center_crop(tensor)?;
            // 应用归一化
            (tensor - &self.mean)?.broadcast_div(&self.std)
        }

        /// 图片裁剪
        pub fn center_crop(&self, img: Tensor) -> Result<Tensor> {
            let (_, h, w) = img.dims3()?;
            let top = (h - self.config.pretrained_cfg.img_size) / 2;
            let left = (w - self.config.pretrained_cfg.img_size) / 2;
            img.narrow(1, top, self.config.pretrained_cfg.img_size)?
                .narrow(2, left, self.config.pretrained_cfg.img_size)
        }
    }

    // 更新后的ViT模型
    pub struct PrithviViT {
        patch_embed: Linear,
        pos_embed: Tensor,
        encoder_layers: Vec<EncoderBlock>,
        decoder: VitDecoder,
        mask_ratio: f32,
        num_features: usize,
    }

    impl PrithviViT {
        pub fn from_pretrained(config: &PrithviConfig, path: &Path, device: &Device) -> Result<Self> {
            let mut varmap = VarMap::new();
            varmap.load(path)?;

            let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

            let model = Self::new(
                vb,
                config,
            )?;

            Ok(model)
        }

        fn new(
            vb: VarBuilder,
            config: &PrithviConfig
        ) -> Result<Self> {
            let num_patches = (config.pretrained_cfg.img_size / config.pretrained_cfg.patch_size[1]) * (config.pretrained_cfg.img_size / config.pretrained_cfg.patch_size[2]);
            let patch_dim = config.pretrained_cfg.in_chans * config.pretrained_cfg.patch_size[0] * config.pretrained_cfg.patch_size[1] * config.pretrained_cfg.patch_size[2];

            // Encoder部分
            let patch_embed = linear(patch_dim, config.pretrained_cfg.embed_dim, vb.pp("patch_embed"))?;
            let pos_embed = vb.get((1, num_patches + 1, config.pretrained_cfg.embed_dim), "pos_embed")?;

            let mut encoder_layers = Vec::with_capacity(config.pretrained_cfg.depth);
            for i in 0..config.pretrained_cfg.depth {
                encoder_layers.push(EncoderBlock::new(
                    vb.pp(&format!("blocks.{}", i)),
                    config.pretrained_cfg.embed_dim,
                    config.pretrained_cfg.num_heads,
                    config.pretrained_cfg.mlp_ratio,
                )?);
            }

            // Decoder部分
            let decoder = VitDecoder::new(
                vb.pp("decoder"),
                config.pretrained_cfg.decoder_embed_dim,
                config.pretrained_cfg.decoder_depth,
                config.pretrained_cfg.decoder_num_heads,
                config.pretrained_cfg.mlp_ratio,
                patch_dim,
            )?;

            Ok(Self {
                patch_embed,
                pos_embed,
                encoder_layers,
                decoder,
                mask_ratio: config.pretrained_cfg.mask_ratio,
                num_features: config.pretrained_cfg.embed_dim,
            })
        }

        fn random_masking(&self, x: &Tensor) -> Result<(Tensor,Tensor)> {
            let (n, l, d) = x.dims3()?;
            let len_keep = (l as f32 * (1.0 - self.mask_ratio)) as usize;

            let noise = Tensor::rand(0.0, 1.0, (n, l), &x.device())?;
            let ids_shuffle = noise.arg_sort_last_dim(true)?;
            let ids_restore = ids_shuffle.arg_sort_last_dim(true)?;

            let x_masked = x.gather(&ids_shuffle.narrow(1, 0, len_keep)?, 1)?;

            Ok((x_masked, ids_restore))
        }

        pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
            // 分块和嵌入
            let (b, c, h, w) = x.dims4()?;
            let x = self.patch_embed.forward(&self.patchify(x)?)?;

            // 添加位置编码
            let x = x.broadcast_add(&self.pos_embed)?;

            // 应用随机mask
            let (x, ids_restore) = self.random_masking(&x)?;

            // 编码器处理
            let mut x = x.clone();
            for layer in &self.encoder_layers {
                x = layer.forward(&x)?;
            }

            // 解码器处理
            let x = self.decoder.forward(&x, &ids_restore)?;

            // 重构输出
            self.unpatchify(&x, h, w)
        }

        fn patchify(&self, x: &Tensor) -> Result<Tensor> {
            let (b, c, h, w) = x.dims4()?;
            let p_t = 1;  // 时间维度分块大小
            let p_h = 16;
            let p_w = 16;

            x.reshape((b, c, h / p_h, p_h, w / p_w, p_w))?
                .permute((0, 2, 4, 3, 5, 1))?
                .reshape((b, (h/p_h)*(w/p_w), p_h*p_w*c))
        }

        fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
            let p = 16;
            x.reshape((x.dim(0)?, h/p, w/p, p*p*6))?
                .permute((0, 3, 1, 2))?
                .reshape((x.dim(0)?, 6, h, w))
        }
    }

    // 解码器实现
    struct VitDecoder {
        blocks: Vec<DecoderBlock>,
        norm: candle_nn::LayerNorm,
        proj: Linear,
    }

    impl VitDecoder {
        fn new(
            vb: VarBuilder,
            embed_dim: usize,
            depth: usize,
            num_heads: usize,
            mlp_ratio: f32,
            output_dim: usize,
        ) -> Result<Self> {
            let mut blocks = Vec::with_capacity(depth);
            for i in 0..depth {
                blocks.push(DecoderBlock::new(
                    embed_dim,
                    output_dim,
                    num_heads,
                    vb.pp(&format!("decoder_blocks.{}", i))
                )?);
            }

            let norm = candle_nn::layer_norm(embed_dim, 1e-6, vb.pp("decoder_norm"))?;
            let proj = linear(embed_dim, output_dim, vb.pp("proj"))?;

            Ok(Self { blocks, norm, proj })
        }

        fn forward(&self, x: &Tensor, ids_restore: &Tensor) -> Result<Tensor> {
            let mut x = x.clone();
            for block in &self.blocks {
                x = block.forward(&x)?;
            }
            x = self.norm.forward(&x)?;

            // 恢复mask tokens
            let mask_tokens = Tensor::zeros(
                (x.dim(0)?, ids_restore.dim(1)? - x.dim(1)?, x.dim(2)?),
                x.dtype(),
                x.device()
            )?;

            let x = Tensor::cat(&[x, mask_tokens], 1)?;
            let x = x.gather(&ids_restore.arg_sort_last_dim(true)?, 1)?;

            self.proj.forward(&x)
        }
    }

    // 更新后的EncoderBlock（添加MLP ratio支持）
    struct EncoderBlock {
        attn: MultiHeadAttention,
        mlp: MLP,
        norm1: candle_nn::LayerNorm,
        norm2: candle_nn::LayerNorm,
    }

    impl EncoderBlock {
        fn new(
            vb: VarBuilder,
            dim: usize,
            num_heads: usize,
            mlp_ratio: f32,
        ) -> Result<Self> {
            let attn = MultiHeadAttention::new(vb.pp("attn"), dim, num_heads)?;
            let mlp = MLP::new(vb.pp("mlp"), dim, mlp_ratio)?;
            let norm1 = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm1"))?;
            let norm2 = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm2"))?;

            Ok(Self { attn, mlp, norm1, norm2 })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let x = (x.clone() + self.attn.forward(&self.norm1.forward(&x.clone())?)?)?;
            let x = (x.clone() + self.mlp.forward(&self.norm2.forward(&x.clone())?)?)?;
            Ok(x)
        }
    }

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

    // 更新后的MLP（支持MLP ratio）
    struct MLP {
        fc1: Linear,
        fc2: Linear,
    }

    impl MLP {
        fn new(vb: VarBuilder, dim: usize, mlp_ratio: f32) -> Result<Self> {
            let hidden_dim = (dim as f32 * mlp_ratio) as usize;
            let fc1 = linear(dim, hidden_dim, vb.pp("fc1"))?;
            let fc2 = linear(hidden_dim, dim, vb.pp("fc2"))?;
            Ok(Self { fc1, fc2 })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
        }
    }

    pub const MODEL_PATH: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/Prithvi_EO_V2_300M_TL.safetensors";
    pub const DATASETS_PATH: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska";
    pub const CONFIG_PATH: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/config.json";

    pub const DATASETS_PATH_TRAIN: &'static str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/florida";

    #[cfg(test)]
    mod tests {
        use candle_transformers::models::llava::config;
        use crate::require_dir_tifs;
        use super::*;

        #[test]
        fn test_model_loading() -> Result<()> {
            let device = Device::Cpu;
            let config = PrithviConfig::from_json(CONFIG_PATH);
            let model = PrithviViT::from_pretrained(&config,Path::new(MODEL_PATH), &device)?;

            assert_eq!(model.num_features, 1024);
            Ok(())
        }

        #[test]
        fn test_forward_pass() -> Result<()> {
            let device = Device::Cpu;
            let config = PrithviConfig::from_json(CONFIG_PATH);
            let model = PrithviViT::from_pretrained(&config, Path::new(MODEL_PATH), &device)?;
            let loader = GeoTiffLoader::new(device.clone(), &config);
            let tifs = require_dir_tifs(DATASETS_PATH)?;
            for tif in tifs {
                let path = std::path::Path::new(&tif);
                let tensor = GeoTiffLoader::load(&loader, path)?.unsqueeze(0)?;
                let output = model.forward(&tensor)?;

                assert_eq!(output.dims(), [1, 6, 224, 224]);
            }
            Ok(())
        }

        #[test]
        fn test_model() -> Result<()> {
            use crate::geotiff_candle_demo::prithvi::{GeoTiffLoader, PrithviViT};
            use candle_core::Device;

            let config = PrithviConfig::from_json(CONFIG_PATH);

            let device = Device::cuda_if_available(0)?;
            let loader = GeoTiffLoader::new(device.clone(), &config);

            let tifs = require_dir_tifs(DATASETS_PATH)?;
            for tif in tifs {
                let path = std::path::PathBuf::from(&tif);
                // 加载数据
                let tensor = loader.load(&path)?;

                // 加载预训练模型
                let model = PrithviViT::from_pretrained(&config, Path::new(MODEL_PATH), &device)?;

                // 前向传播
                let output = model.forward(&tensor.unsqueeze(0)?)?;

                println!("Reconstructed output shape: {:?}", output.dims());
            }

            Ok(())
        }
    }
}