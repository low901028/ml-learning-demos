
use std::collections::HashMap;
use std::fs::File;
// 三维ViT模型实现
use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_core::op::UnaryOp::Gelu;
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, Module, linear, layer_norm, LayerNorm, Linear, conv2d};
use tiff::{decoder::Decoder as TiffDecoder, ColorType};
use candle_core::safetensors::MmapedSafetensors;
use tiff::decoder::DecodingResult;
use tiff::encoder::TiffValue;
use crate::geotiff_candle_demo::prithvi::{CONFIG_PATH, DATASETS_PATH_TRAIN};
use crate::{require_dir_tifs, MODEL_PATH};

/// =======================================
/// config.json文件
/// =======================================
#[derive(Debug, Clone,PartialEq, serde::Deserialize, serde::Serialize, Default)]
pub struct PretrainedConfig {
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
    pub pretrained_cfg: PretrainedConfig,
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

/// =================================
/// 3D卷积实现
/// =================================
// 1. 添加Conv3D配置结构
#[derive(Debug, Clone, Copy)]
pub struct Conv3dConfig {
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
        }
    }
}

// 2. 实现Conv3D结构
pub struct Conv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv3dConfig,
}

impl Conv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        cfg: Conv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        // 参数校验
        if in_channels % cfg.groups != 0 {
            candle_core::bail!(
                "Input channels {} must be divisible by groups {}",
                in_channels,
                cfg.groups
            );
        }

        let weight_shape = (
            out_channels,
            in_channels / cfg.groups,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
        );

        // 使用更安全的权重初始化方式
        let weight = vb.get_with_hints(
            weight_shape,
            "weight",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;

        // 支持可选偏置项
        let bias = vb.get(out_channels, "bias")?;

        Ok(Self {
            weight,
            bias: Some(bias),
            config: cfg,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, in_c, t, h, w) = x.dims5()?;
        let (out_c, _, kt, kh, kw) = self.weight.dims5()?;

        // 使用Candle内置的conv3d实现
        let x = conv3d_ops(
            x,
            &self.weight,
            self.config.stride,
            self.config.padding,
            self.config.dilation,
            self.config.groups,
        )?;

        // 处理偏置项
        match &self.bias {
            Some(bias) => x.broadcast_add(bias),
            None => Ok(x),
        }
    }
}

// 在candle_nn::ops模块中添加conv3d实现
pub fn conv3d_ops(
        x: &Tensor,
        weight: &Tensor,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Result<Tensor> {
        // 实现基于im2col算法的3D卷积
        let (b, in_c, t, h, w) = x.dims5()?;
        let (out_c, in_c_g, kt, kh, kw) = weight.dims5()?;

        // 计算输出形状
        let t_out = (t + 2*padding.0 - dilation.0*(kt - 1) - 1)/stride.0 + 1;
        let h_out = (h + 2*padding.1 - dilation.1*(kh - 1) - 1)/stride.1 + 1;
        let w_out = (w + 2*padding.2 - dilation.2*(kw - 1) - 1)/stride.2 + 1;

        // 创建输出张量
        let dev = x.device();
        let mut output = Tensor::zeros(
            Shape::from(&[b, out_c, t_out, h_out, w_out]),
            DType::F32,
            dev,
        )?;

        // 实现卷积核
        for b_idx in 0..b {
            for oc in 0..out_c {
                for t_out_idx in 0..t_out {
                    for h_out_idx in 0..h_out {
                        for w_out_idx in 0..w_out {
                            // 计算输入窗口位置
                            let t_start = t_out_idx * stride.0;
                            let h_start = h_out_idx * stride.1;
                            let w_start = w_out_idx * stride.2;

                            // 提取输入窗口
                            let window = x.narrow(0, b_idx, 1)?
                                .narrow(1, 0, in_c)?
                                .narrow(2, t_start, kt)?
                                .narrow(3, h_start, kh)?
                                .narrow(4, w_start, kw)?;

                            // 计算卷积结果
                            let weight_slice = weight.narrow(0, oc, 1)?;
                            let conv_result = window
                                .mul(&weight_slice)?
                                .sum_all()?;
                            // 存储结果
                            output = output.slice_assign(
                                &[
                                    b_idx..b_idx+1,
                                    oc..oc+1,
                                    t_out_idx..t_out_idx+1,
                                    h_out_idx..h_out_idx+1,
                                    w_out_idx..w_out_idx+1,
                                ],
                                &conv_result,
                            )?;
                        }
                    }
                }
            }
        }

        Ok(output)
}

// 3. 创建conv3d函数（类似candle_nn::conv2d）
pub fn conv3d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    cfg: Conv3dConfig,
    vb: VarBuilder,
) -> Result<Conv3d> {
    Conv3d::new(in_channels, out_channels, kernel_size, cfg, vb)
}

// 4. 修改PatchEmbed3D结构
struct PatchEmbed3D {
    conv: Conv3d,
    time_dim: usize,
}

impl PatchEmbed3D {
    fn new(config: &PretrainedConfig, vb: VarBuilder) -> Result<Self> {
        let (t_patch, h_patch, w_patch) = (
            config.patch_size[0],
            config.patch_size[1],
            config.patch_size[2],
        );

        let conv_cfg = Conv3dConfig {
            stride: (t_patch, h_patch, w_patch),
            padding: (0, 0, 0),
            ..Default::default()
        };

        let conv = conv3d(
            config.in_chans,
            config.embed_dim,
            (t_patch, h_patch, w_patch),
            conv_cfg,
            vb.pp("proj"),
        )?;

        Ok(Self {
            conv,
            time_dim: config.num_frames,
        })
    }
}

impl Module for PatchEmbed3D {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 输入形状应为 [B, C, T, H, W]
        let (b, c, t, h, w) = x.dims5()?;

        // 应用3D卷积
        let x = self.conv.forward(x)?; // 输出形状 [B, D, T', H', W']

        // 调整维度顺序为 [B, T', D, H', W']
        x.permute((0, 2, 1, 3, 4))?.contiguous()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ===============
/// 加载safetensors模型权重
struct ModelLoader {
    device: Device,
}

// 5. 更新模型加载器
impl ModelLoader {
    fn new(device: Device) -> Result<Self> {
        Ok(Self { device })
    }
    fn vb(&self, path: &str) -> Result<VarBuilder> {
        let path = std::path::PathBuf::from(path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device)?
        };
        Ok(vb)
    }

    fn load_3d_conv_weight(&self, name: &str) -> Result<Tensor> {
        let weight = self.vb("/Users/dalan/rustspace/ml-learning-demos/prithvi-model/Prithvi_EO_V2_300M_TL.safetensors")?.get((),name)?;
        // 验证权重形状为5D [out_c, in_c, t, h, w]
        if weight.rank() != 5 {
            candle_core::bail!("Invalid 3D conv weight shape: {:?}", weight.dims());
        }

        // 验证具体维度
        let expected_shape = [1024, 6, 1, 16, 16];
        if weight.dims() != &expected_shape {
            candle_core::bail!(
                "Weight shape mismatch: {:?} vs expected {:?}",
                weight.dims(),
                expected_shape
            );
        }

        Ok(weight)
    }
}

// 更新位置编码实现
struct PositionEmbedding3D {
    pos_embed: Tensor,
}

impl PositionEmbedding3D {
    fn from_tensor(tensor: Tensor) -> Self {
        Self { pos_embed: tensor }
    }
}

// 更新Transformer块结构
struct TransformerBlock {
    norm1: LayerNorm,
    attn: MultiHeadAttention,
    norm2: LayerNorm,
    mlp: MLP,
}

impl TransformerBlock {
    fn new(embed_dim: usize, num_heads: usize, mlp_ratio: f32, vb: VarBuilder) -> Result<Self> {
        let norm1 = layer_norm(embed_dim, 1e-6, vb.pp("norm1"))?;
        let attn = MultiHeadAttention::new(embed_dim, num_heads, vb.pp("attn"))?;
        let norm2 = layer_norm(embed_dim, 1e-6, vb.pp("norm2"))?;
        let mlp = MLP::new(embed_dim, mlp_ratio, vb.pp("mlp"))?;
        Ok(Self { norm1, attn, norm2, mlp })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // 残差连接 + 注意力
        let xs = {
            let normalized = self.norm1.forward(xs)?;
            let attention = self.attn.forward(&normalized)?;
            xs.add(&attention)?
        };

        // 残差连接 + MLP
        let xs = {
            let normalized = self.norm2.forward(&xs)?;
            let mlp_out = self.mlp.forward(&normalized)?;
            xs.add(&mlp_out)?
        };

        Ok(xs)
    }
}

struct MultiHeadAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
}

impl MultiHeadAttention {
    fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let qkv = linear(embed_dim, 3 * embed_dim, vb.pp("qkv"))?;
        let proj = linear(embed_dim, embed_dim, vb.pp("proj"))?;
        Ok(Self { qkv, proj, num_heads })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?; // [batch, seq_len, embed_dim]
        let head_dim = c / self.num_heads;

        // 生成QKV矩阵
        let qkv = self.qkv.forward(xs)?; // [b, n, 3*embed_dim]
        let qkv = qkv.reshape((b, n, 3, self.num_heads, head_dim))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?; // [b, heads, n, head_dim]
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;

        // 计算注意力分数
        let scale = (head_dim as f64).sqrt().recip() as f64;
        let attn = q.matmul(&k.t()?)? * scale; // [b, heads, n, n]
        let attn = candle_nn::ops::softmax(&attn?, 3)?;

        // 应用注意力权重到V
        let mut x = attn.matmul(&v)?; // [b, heads, n, head_dim]
        x = x.transpose(1, 2)?.reshape((b, n, c))?;

        // 最终投影
        self.proj.forward(&x)
    }
}


struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new(embed_dim: usize, mlp_ratio: f32, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (embed_dim as f32 * mlp_ratio) as usize;
        let fc1 = linear(embed_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = linear(hidden_dim, embed_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = &xs.gelu()?; // 使用GELU激活函数
        self.fc2.forward(&xs)
    }
}


// 更新完整模型结构
struct PrithviModel {
    encoder: Encoder,
    decoder: Decoder,
    config: ModelConfig,
}

struct Encoder {
    cls_token: Tensor,
    patch_embed: PatchEmbed3D,
    pos_embed: PositionEmbedding3D,
    blocks: Vec<TransformerBlock>,
    norm: LayerNorm,
    temporal_scale: Tensor,
    location_scale: Tensor,
}

struct Decoder {
    mask_token: Tensor,
    decoder_embed: Linear,
    decoder_pos_embed: PositionEmbedding3D,
    blocks: Vec<TransformerBlock>,
    decoder_norm: LayerNorm,
    decoder_pred: Linear,
}

impl PrithviModel {
    fn new(config: ModelConfig, loader: &ModelLoader) -> Result<Self> {
        let model_loader= ModelLoader::new(Device::Cpu)?;
        let vb = model_loader.vb("/Users/dalan/rustspace/ml-learning-demos/prithvi-model/Prithvi_EO_V2_300M_TL.safetensors")?;
        let pretrained_cfg = &config.pretrained_cfg;

        // 初始化编码器
        // 初始化编码器（关键修改部分）
        let encoder = {
            // 加载CLS token
            let cls_token = vb.get((1, 1, 1024), "encoder.cls_token")?;

            // 初始化3D卷积层并加载权重
            let patch_embed = {
                let vb_conv = vb.pp("encoder.patch_embed");
                let mut conv = PatchEmbed3D::new(pretrained_cfg, vb_conv.clone())?;

                // 验证权重形状
                let weight_shape = conv.conv.weight.dims();
                assert_eq!(
                    weight_shape,
                    &[1024, 6, 1, 16, 16],
                    "Conv3D weight shape mismatch: {:?}",
                    weight_shape
                );
                conv
            };

            // 加载位置编码
            let pos_embed = PositionEmbedding3D::from_tensor(
                vb.get((1, 785, 1024), "encoder.pos_embed")?
            );

            // 初始化Transformer块
            let mut blocks = Vec::new();
            for i in 0..pretrained_cfg.depth {
                blocks.push(TransformerBlock::new(
                    pretrained_cfg.embed_dim,
                    pretrained_cfg.num_heads,
                    pretrained_cfg.mlp_ratio,
                    vb.pp(&format!("encoder.blocks.{}", i)),
                )?);
            }

            Encoder {
                cls_token,
                patch_embed,
                pos_embed,
                blocks,
                norm: layer_norm(pretrained_cfg.embed_dim, 1e-6, vb.pp("encoder.norm"))?,
                temporal_scale: vb.get((1), "encoder.temporal_embed_enc.scale")?,
                location_scale: vb.get((1), "encoder.location_embed_enc.scale")?,
            }
        };


        // 初始化解码器
        let decoder = {
            let mask_token = vb.get((1, 1, 512),"decoder.mask_token")?;
            let decoder_embed = linear(
                pretrained_cfg.embed_dim,
                pretrained_cfg.decoder_embed_dim,
                vb.pp("decoder.decoder_embed"),
            )?;
            let decoder_pos_embed = PositionEmbedding3D::from_tensor(
                vb.get((1, 785, 512),"decoder.decoder_pos_embed")?
            );
            let mut blocks = Vec::new();
            for i in 0..pretrained_cfg.decoder_depth {
                blocks.push(TransformerBlock::new(
                    pretrained_cfg.decoder_embed_dim,
                    pretrained_cfg.decoder_num_heads,
                    pretrained_cfg.mlp_ratio,
                    vb.pp(&format!("decoder.decoder_blocks.{}", i)),
                )?);
            }
            let decoder_norm = layer_norm(
                pretrained_cfg.decoder_embed_dim,
                1e-6,
                vb.pp("decoder.decoder_norm"),
            )?;
            let decoder_pred = linear(
                pretrained_cfg.decoder_embed_dim,
                pretrained_cfg.in_chans * pretrained_cfg.patch_size[1] * pretrained_cfg.patch_size[2],
                vb.pp("decoder.decoder_pred"),
            )?;

            Decoder { mask_token, decoder_embed, decoder_pos_embed, blocks, decoder_norm, decoder_pred }
        };

        Ok(Self { encoder, decoder, config })
    }

    fn forward(&self, x: &Tensor, mask_ratio: f32) -> Result<Tensor> {
        let (b, t, c, h, w) = x.dims5()?;

        // if c != self.config.pretrained_cfg.in_chans {
        //     candle_core::bail!(
        //         "Input channels mismatch: expected {}, got {}",
        //         self.config.pretrained_cfg.in_chans,
        //         c
        //     );
        // }

        // 编码器前向
        let patches = self.encoder.patch_embed.forward(x)?;
        let patches = patches.flatten_from(2)?.flatten(1, 2)?; // [B, T*H*W, D]

        // 添加位置编码
        let pos_embed = self.encoder.pos_embed.pos_embed.unsqueeze(0)?;
        let x = patches.broadcast_add(&pos_embed)?;

        // 添加CLS token
        let cls_tokens = self.encoder.cls_token.expand((b, 1, self.config.pretrained_cfg.embed_dim))?;
        let x = Tensor::cat(&[cls_tokens, x], 1)?;

        // 通过Transformer编码器
        let mut x = x;
        for block in &self.encoder.blocks {
            x = block.forward(&x)?;
        }
        x = self.encoder.norm.forward(&x)?;

        // 解码器前向
        let x = self.decoder.decoder_embed.forward(&x)?;
        let pos_embed_dec = self.decoder.decoder_pos_embed.pos_embed.unsqueeze(0)?;
        let x = x.broadcast_add(&pos_embed_dec)?;

        let x = self.decoder.blocks.iter().try_fold(x, |x, block| {
            block.forward(&x)
        })?;

        let x = self.decoder.decoder_norm.forward(&x)?;
        let x = self.decoder.decoder_pred.forward(&x)?;

        Ok(x)
    }
}

// 测试用例
#[cfg(test)]
mod tests {
    use candle_core::safetensors::load;
    use crate::geotiff_candle_demo::prithvi::CONFIG_PATH;
    use super::*;

    #[test]
    fn test_parameter_loading() -> Result<()> {
        let dev = Device::Cpu;
        let loader = ModelLoader::new(Device::Cpu)?;
        let mut vb = loader.vb("/Users/dalan/rustspace/ml-learning-demos/prithvi-model/Prithvi_EO_V2_300M_TL.safetensors")?;
        // 验证关键参数形状
        let cls_token = vb.get((1, 1, 1024), "encoder.cls_token")?;
        assert_eq!(cls_token.dims(), &[1, 1, 1024]);

        let patch_weight = vb.get((1024, 6, 1, 16, 16),"encoder.patch_embed.proj.weight")?;
        assert_eq!(patch_weight.dims(), &[1024, 6, 1, 16, 16]);

        let qkv_weight = vb.get((3072, 1024),"encoder.blocks.0.attn.qkv.weight")?;
        assert_eq!(qkv_weight.dims(), &[3072, 1024]);

        Ok(())
    }

    #[test]
    fn test_full_forward() -> Result<()> {
        let dev = Device::Cpu;
        let config = ModelConfig::from_json(CONFIG_PATH);
        let loader = ModelLoader::new(dev.clone())?;
        let model = PrithviModel::new(config, &loader)?;

        // 生成符合3D卷积输入的张量 [B, C, T, H, W]
        let input = Tensor::randn(0f32, 1f32, (2, 6, 4, 224, 224), &dev)?;

        // 执行前向传播
        let output = model.forward(&input, 0.75)?;

        // 验证输出形状
        let expected_shape = [
            2,
            1 + 4*(224/16)*(224/16), // CLS token + patches
            6*16*16 // 重建像素
        ];
        assert_eq!(output.dims(), &expected_shape);
        Ok(())
    }

    #[test]
    fn test_3d_conv() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);

        // 测试3D卷积层
        let conv3d = Conv3d::new(
            6,  // 输入通道
            1024, // 输出通道
            (1, 16, 16), // 3D卷积核
            Conv3dConfig {
                stride: (1, 16, 16),
                ..Default::default()
            },
            vb.pp("test_conv3d"),
        )?;

        // 验证权重形状
        assert_eq!(conv3d.weight.dims(), &[1024, 6, 1, 16, 16]);

        // 验证前向传播
        let input = Tensor::randn(0f32, 1f32, (2, 6, 4, 224, 224), &dev)?;
        let output = conv3d.forward(&input)?;
        assert_eq!(output.dims(), &[2, 1024, 4, 14, 14]); // 224/16=14

        Ok(())
    }
}

// 主函数示例
#[test]
fn main() -> Result<()> {
    let dev = Device::Cpu;
    let config = ModelConfig::from_json(CONFIG_PATH);
    let loader = ModelLoader::new(dev.clone())?;
    let model = PrithviModel::new(config, &loader)?;

    // 加载输入数据
    let tifs = require_dir_tifs(DATASETS_PATH_TRAIN)?;
    for tiff in tifs {
        // let input = load_tiff_as_tensor(&tiff, &dev)?;
        let input  = load_geotiff_as_tensor(&tiff, &dev)?;
        // 执行推理
        let output = model.forward(&input, 0.0)?; // mask_ratio=0表示全量预测
        println!("Reconstruction output shape: {:?}", output.dims());
    }

    Ok(())
}

#[test]
fn get_safetensors_file_path() -> Result<()> {
    let weight_params = vec!["encoder.cls_token",
                             "decoder.mask_token",
                             "encoder.pos_embed",
                             "decoder.decoder_pos_embed",
                             "encoder.patch_embed.proj.weight",
                             "encoder.patch_embed.proj.bias",
                             "encoder.temporal_embed_enc.scale",
                             "encoder.location_embed_enc.scale",
                             "encoder.blocks.0.norm1.weight",
                             "encoder.blocks.0.norm1.bias",
                             "encoder.blocks.0.attn.qkv.weight",
                             "encoder.blocks.0.attn.qkv.bias",
                             "encoder.blocks.0.attn.proj.weight",
                             "encoder.blocks.0.attn.proj.bias",
                             "encoder.blocks.0.norm2.weight",
                             "encoder.blocks.0.norm2.bias",
                             "encoder.blocks.0.mlp.fc1.weight",
                             "encoder.blocks.0.mlp.fc1.bias",
                             "encoder.blocks.0.mlp.fc2.weight",
                             "encoder.blocks.0.mlp.fc2.bias",
                             "encoder.blocks.1.norm1.weight",
                             "encoder.blocks.1.norm1.bias",
                             "encoder.blocks.1.attn.qkv.weight",
                             "encoder.blocks.1.attn.qkv.bias",
                             "encoder.blocks.1.attn.proj.weight",
                             "encoder.blocks.1.attn.proj.bias",
                             "encoder.blocks.1.norm2.weight",
                             "encoder.blocks.1.norm2.bias",
                             "encoder.blocks.1.mlp.fc1.weight",
                             "encoder.blocks.1.mlp.fc1.bias",
                             "encoder.blocks.1.mlp.fc2.weight",
                             "encoder.blocks.1.mlp.fc2.bias",
                             "encoder.blocks.2.norm1.weight",
                             "encoder.blocks.2.norm1.bias",
                             "encoder.blocks.2.attn.qkv.weight",
                             "encoder.blocks.2.attn.qkv.bias",
                             "encoder.blocks.2.attn.proj.weight",
                             "encoder.blocks.2.attn.proj.bias",
                             "encoder.blocks.2.norm2.weight",
                             "encoder.blocks.2.norm2.bias",
                             "encoder.blocks.2.mlp.fc1.weight",
                             "encoder.blocks.2.mlp.fc1.bias",
                             "encoder.blocks.2.mlp.fc2.weight",
                             "encoder.blocks.2.mlp.fc2.bias",
                             "encoder.blocks.3.norm1.weight",
                             "encoder.blocks.3.norm1.bias",
                             "encoder.blocks.3.attn.qkv.weight",
                             "encoder.blocks.3.attn.qkv.bias",
                             "encoder.blocks.3.attn.proj.weight",
                             "encoder.blocks.3.attn.proj.bias",
                             "encoder.blocks.3.norm2.weight",
                             "encoder.blocks.3.norm2.bias",
                             "encoder.blocks.3.mlp.fc1.weight",
                             "encoder.blocks.3.mlp.fc1.bias",
                             "encoder.blocks.3.mlp.fc2.weight",
                             "encoder.blocks.3.mlp.fc2.bias",
                             "encoder.blocks.4.norm1.weight",
                             "encoder.blocks.4.norm1.bias",
                             "encoder.blocks.4.attn.qkv.weight",
                             "encoder.blocks.4.attn.qkv.bias",
                             "encoder.blocks.4.attn.proj.weight",
                             "encoder.blocks.4.attn.proj.bias",
                             "encoder.blocks.4.norm2.weight",
                             "encoder.blocks.4.norm2.bias",
                             "encoder.blocks.4.mlp.fc1.weight",
                             "encoder.blocks.4.mlp.fc1.bias",
                             "encoder.blocks.4.mlp.fc2.weight",
                             "encoder.blocks.4.mlp.fc2.bias",
                             "encoder.blocks.5.norm1.weight",
                             "encoder.blocks.5.norm1.bias",
                             "encoder.blocks.5.attn.qkv.weight",
                             "encoder.blocks.5.attn.qkv.bias",
                             "encoder.blocks.5.attn.proj.weight",
                             "encoder.blocks.5.attn.proj.bias",
                             "encoder.blocks.5.norm2.weight",
                             "encoder.blocks.5.norm2.bias",
                             "encoder.blocks.5.mlp.fc1.weight",
                             "encoder.blocks.5.mlp.fc1.bias",
                             "encoder.blocks.5.mlp.fc2.weight",
                             "encoder.blocks.5.mlp.fc2.bias",
                             "encoder.blocks.6.norm1.weight",
                             "encoder.blocks.6.norm1.bias",
                             "encoder.blocks.6.attn.qkv.weight",
                             "encoder.blocks.6.attn.qkv.bias",
                             "encoder.blocks.6.attn.proj.weight",
                             "encoder.blocks.6.attn.proj.bias",
                             "encoder.blocks.6.norm2.weight",
                             "encoder.blocks.6.norm2.bias",
                             "encoder.blocks.6.mlp.fc1.weight",
                             "encoder.blocks.6.mlp.fc1.bias",
                             "encoder.blocks.6.mlp.fc2.weight",
                             "encoder.blocks.6.mlp.fc2.bias",
                             "encoder.blocks.7.norm1.weight",
                             "encoder.blocks.7.norm1.bias",
                             "encoder.blocks.7.attn.qkv.weight",
                             "encoder.blocks.7.attn.qkv.bias",
                             "encoder.blocks.7.attn.proj.weight",
                             "encoder.blocks.7.attn.proj.bias",
                             "encoder.blocks.7.norm2.weight",
                             "encoder.blocks.7.norm2.bias",
                             "encoder.blocks.7.mlp.fc1.weight",
                             "encoder.blocks.7.mlp.fc1.bias",
                             "encoder.blocks.7.mlp.fc2.weight",
                             "encoder.blocks.7.mlp.fc2.bias",
                             "encoder.blocks.8.norm1.weight",
                             "encoder.blocks.8.norm1.bias",
                             "encoder.blocks.8.attn.qkv.weight",
                             "encoder.blocks.8.attn.qkv.bias",
                             "encoder.blocks.8.attn.proj.weight",
                             "encoder.blocks.8.attn.proj.bias",
                             "encoder.blocks.8.norm2.weight",
                             "encoder.blocks.8.norm2.bias",
                             "encoder.blocks.8.mlp.fc1.weight",
                             "encoder.blocks.8.mlp.fc1.bias",
                             "encoder.blocks.8.mlp.fc2.weight",
                             "encoder.blocks.8.mlp.fc2.bias",
                             "encoder.blocks.9.norm1.weight",
                             "encoder.blocks.9.norm1.bias",
                             "encoder.blocks.9.attn.qkv.weight",
                             "encoder.blocks.9.attn.qkv.bias",
                             "encoder.blocks.9.attn.proj.weight",
                             "encoder.blocks.9.attn.proj.bias",
                             "encoder.blocks.9.norm2.weight",
                             "encoder.blocks.9.norm2.bias",
                             "encoder.blocks.9.mlp.fc1.weight",
                             "encoder.blocks.9.mlp.fc1.bias",
                             "encoder.blocks.9.mlp.fc2.weight",
                             "encoder.blocks.9.mlp.fc2.bias",
                             "encoder.blocks.10.norm1.weight",
                             "encoder.blocks.10.norm1.bias",
                             "encoder.blocks.10.attn.qkv.weight",
                             "encoder.blocks.10.attn.qkv.bias",
                             "encoder.blocks.10.attn.proj.weight",
                             "encoder.blocks.10.attn.proj.bias",
                             "encoder.blocks.10.norm2.weight",
                             "encoder.blocks.10.norm2.bias",
                             "encoder.blocks.10.mlp.fc1.weight",
                             "encoder.blocks.10.mlp.fc1.bias",
                             "encoder.blocks.10.mlp.fc2.weight",
                             "encoder.blocks.10.mlp.fc2.bias",
                             "encoder.blocks.11.norm1.weight",
                             "encoder.blocks.11.norm1.bias",
                             "encoder.blocks.11.attn.qkv.weight",
                             "encoder.blocks.11.attn.qkv.bias",
                             "encoder.blocks.11.attn.proj.weight",
                             "encoder.blocks.11.attn.proj.bias",
                             "encoder.blocks.11.norm2.weight",
                             "encoder.blocks.11.norm2.bias",
                             "encoder.blocks.11.mlp.fc1.weight",
                             "encoder.blocks.11.mlp.fc1.bias",
                             "encoder.blocks.11.mlp.fc2.weight",
                             "encoder.blocks.11.mlp.fc2.bias",
                             "encoder.blocks.12.norm1.weight",
                             "encoder.blocks.12.norm1.bias",
                             "encoder.blocks.12.attn.qkv.weight",
                             "encoder.blocks.12.attn.qkv.bias",
                             "encoder.blocks.12.attn.proj.weight",
                             "encoder.blocks.12.attn.proj.bias",
                             "encoder.blocks.12.norm2.weight",
                             "encoder.blocks.12.norm2.bias",
                             "encoder.blocks.12.mlp.fc1.weight",
                             "encoder.blocks.12.mlp.fc1.bias",
                             "encoder.blocks.12.mlp.fc2.weight",
                             "encoder.blocks.12.mlp.fc2.bias",
                             "encoder.blocks.13.norm1.weight",
                             "encoder.blocks.13.norm1.bias",
                             "encoder.blocks.13.attn.qkv.weight",
                             "encoder.blocks.13.attn.qkv.bias",
                             "encoder.blocks.13.attn.proj.weight",
                             "encoder.blocks.13.attn.proj.bias",
                             "encoder.blocks.13.norm2.weight",
                             "encoder.blocks.13.norm2.bias",
                             "encoder.blocks.13.mlp.fc1.weight",
                             "encoder.blocks.13.mlp.fc1.bias",
                             "encoder.blocks.13.mlp.fc2.weight",
                             "encoder.blocks.13.mlp.fc2.bias",
                             "encoder.blocks.14.norm1.weight",
                             "encoder.blocks.14.norm1.bias",
                             "encoder.blocks.14.attn.qkv.weight",
                             "encoder.blocks.14.attn.qkv.bias",
                             "encoder.blocks.14.attn.proj.weight",
                             "encoder.blocks.14.attn.proj.bias",
                             "encoder.blocks.14.norm2.weight",
                             "encoder.blocks.14.norm2.bias",
                             "encoder.blocks.14.mlp.fc1.weight",
                             "encoder.blocks.14.mlp.fc1.bias",
                             "encoder.blocks.14.mlp.fc2.weight",
                             "encoder.blocks.14.mlp.fc2.bias",
                             "encoder.blocks.15.norm1.weight",
                             "encoder.blocks.15.norm1.bias",
                             "encoder.blocks.15.attn.qkv.weight",
                             "encoder.blocks.15.attn.qkv.bias",
                             "encoder.blocks.15.attn.proj.weight",
                             "encoder.blocks.15.attn.proj.bias",
                             "encoder.blocks.15.norm2.weight",
                             "encoder.blocks.15.norm2.bias",
                             "encoder.blocks.15.mlp.fc1.weight",
                             "encoder.blocks.15.mlp.fc1.bias",
                             "encoder.blocks.15.mlp.fc2.weight",
                             "encoder.blocks.15.mlp.fc2.bias",
                             "encoder.blocks.16.norm1.weight",
                             "encoder.blocks.16.norm1.bias",
                             "encoder.blocks.16.attn.qkv.weight",
                             "encoder.blocks.16.attn.qkv.bias",
                             "encoder.blocks.16.attn.proj.weight",
                             "encoder.blocks.16.attn.proj.bias",
                             "encoder.blocks.16.norm2.weight",
                             "encoder.blocks.16.norm2.bias",
                             "encoder.blocks.16.mlp.fc1.weight",
                             "encoder.blocks.16.mlp.fc1.bias",
                             "encoder.blocks.16.mlp.fc2.weight",
                             "encoder.blocks.16.mlp.fc2.bias",
                             "encoder.blocks.17.norm1.weight",
                             "encoder.blocks.17.norm1.bias",
                             "encoder.blocks.17.attn.qkv.weight",
                             "encoder.blocks.17.attn.qkv.bias",
                             "encoder.blocks.17.attn.proj.weight",
                             "encoder.blocks.17.attn.proj.bias",
                             "encoder.blocks.17.norm2.weight",
                             "encoder.blocks.17.norm2.bias",
                             "encoder.blocks.17.mlp.fc1.weight",
                             "encoder.blocks.17.mlp.fc1.bias",
                             "encoder.blocks.17.mlp.fc2.weight",
                             "encoder.blocks.17.mlp.fc2.bias",
                             "encoder.blocks.18.norm1.weight",
                             "encoder.blocks.18.norm1.bias",
                             "encoder.blocks.18.attn.qkv.weight",
                             "encoder.blocks.18.attn.qkv.bias",
                             "encoder.blocks.18.attn.proj.weight",
                             "encoder.blocks.18.attn.proj.bias",
                             "encoder.blocks.18.norm2.weight",
                             "encoder.blocks.18.norm2.bias",
                             "encoder.blocks.18.mlp.fc1.weight",
                             "encoder.blocks.18.mlp.fc1.bias",
                             "encoder.blocks.18.mlp.fc2.weight",
                             "encoder.blocks.18.mlp.fc2.bias",
                             "encoder.blocks.19.norm1.weight",
                             "encoder.blocks.19.norm1.bias",
                             "encoder.blocks.19.attn.qkv.weight",
                             "encoder.blocks.19.attn.qkv.bias",
                             "encoder.blocks.19.attn.proj.weight",
                             "encoder.blocks.19.attn.proj.bias",
                             "encoder.blocks.19.norm2.weight",
                             "encoder.blocks.19.norm2.bias",
                             "encoder.blocks.19.mlp.fc1.weight",
                             "encoder.blocks.19.mlp.fc1.bias",
                             "encoder.blocks.19.mlp.fc2.weight",
                             "encoder.blocks.19.mlp.fc2.bias",
                             "encoder.blocks.20.norm1.weight",
                             "encoder.blocks.20.norm1.bias",
                             "encoder.blocks.20.attn.qkv.weight",
                             "encoder.blocks.20.attn.qkv.bias",
                             "encoder.blocks.20.attn.proj.weight",
                             "encoder.blocks.20.attn.proj.bias",
                             "encoder.blocks.20.norm2.weight",
                             "encoder.blocks.20.norm2.bias",
                             "encoder.blocks.20.mlp.fc1.weight",
                             "encoder.blocks.20.mlp.fc1.bias",
                             "encoder.blocks.20.mlp.fc2.weight",
                             "encoder.blocks.20.mlp.fc2.bias",
                             "encoder.blocks.21.norm1.weight",
                             "encoder.blocks.21.norm1.bias",
                             "encoder.blocks.21.attn.qkv.weight",
                             "encoder.blocks.21.attn.qkv.bias",
                             "encoder.blocks.21.attn.proj.weight",
                             "encoder.blocks.21.attn.proj.bias",
                             "encoder.blocks.21.norm2.weight",
                             "encoder.blocks.21.norm2.bias",
                             "encoder.blocks.21.mlp.fc1.weight",
                             "encoder.blocks.21.mlp.fc1.bias",
                             "encoder.blocks.21.mlp.fc2.weight",
                             "encoder.blocks.21.mlp.fc2.bias",
                             "encoder.blocks.22.norm1.weight",
                             "encoder.blocks.22.norm1.bias",
                             "encoder.blocks.22.attn.qkv.weight",
                             "encoder.blocks.22.attn.qkv.bias",
                             "encoder.blocks.22.attn.proj.weight",
                             "encoder.blocks.22.attn.proj.bias",
                             "encoder.blocks.22.norm2.weight",
                             "encoder.blocks.22.norm2.bias",
                             "encoder.blocks.22.mlp.fc1.weight",
                             "encoder.blocks.22.mlp.fc1.bias",
                             "encoder.blocks.22.mlp.fc2.weight",
                             "encoder.blocks.22.mlp.fc2.bias",
                             "encoder.blocks.23.norm1.weight",
                             "encoder.blocks.23.norm1.bias",
                             "encoder.blocks.23.attn.qkv.weight",
                             "encoder.blocks.23.attn.qkv.bias",
                             "encoder.blocks.23.attn.proj.weight",
                             "encoder.blocks.23.attn.proj.bias",
                             "encoder.blocks.23.norm2.weight",
                             "encoder.blocks.23.norm2.bias",
                             "encoder.blocks.23.mlp.fc1.weight",
                             "encoder.blocks.23.mlp.fc1.bias",
                             "encoder.blocks.23.mlp.fc2.weight",
                             "encoder.blocks.23.mlp.fc2.bias",
                             "encoder.norm.weight",
                             "encoder.norm.bias",
                             "decoder.decoder_embed.weight",
                             "decoder.decoder_embed.bias",
                             "decoder.temporal_embed_dec.scale",
                             "decoder.location_embed_dec.scale",
                             "decoder.decoder_blocks.0.norm1.weight",
                             "decoder.decoder_blocks.0.norm1.bias",
                             "decoder.decoder_blocks.0.attn.qkv.weight",
                             "decoder.decoder_blocks.0.attn.qkv.bias",
                             "decoder.decoder_blocks.0.attn.proj.weight",
                             "decoder.decoder_blocks.0.attn.proj.bias",
                             "decoder.decoder_blocks.0.norm2.weight",
                             "decoder.decoder_blocks.0.norm2.bias",
                             "decoder.decoder_blocks.0.mlp.fc1.weight",
                             "decoder.decoder_blocks.0.mlp.fc1.bias",
                             "decoder.decoder_blocks.0.mlp.fc2.weight",
                             "decoder.decoder_blocks.0.mlp.fc2.bias",
                             "decoder.decoder_blocks.1.norm1.weight",
                             "decoder.decoder_blocks.1.norm1.bias",
                             "decoder.decoder_blocks.1.attn.qkv.weight",
                             "decoder.decoder_blocks.1.attn.qkv.bias",
                             "decoder.decoder_blocks.1.attn.proj.weight",
                             "decoder.decoder_blocks.1.attn.proj.bias",
                             "decoder.decoder_blocks.1.norm2.weight",
                             "decoder.decoder_blocks.1.norm2.bias",
                             "decoder.decoder_blocks.1.mlp.fc1.weight",
                             "decoder.decoder_blocks.1.mlp.fc1.bias",
                             "decoder.decoder_blocks.1.mlp.fc2.weight",
                             "decoder.decoder_blocks.1.mlp.fc2.bias",
                             "decoder.decoder_blocks.2.norm1.weight",
                             "decoder.decoder_blocks.2.norm1.bias",
                             "decoder.decoder_blocks.2.attn.qkv.weight",
                             "decoder.decoder_blocks.2.attn.qkv.bias",
                             "decoder.decoder_blocks.2.attn.proj.weight",
                             "decoder.decoder_blocks.2.attn.proj.bias",
                             "decoder.decoder_blocks.2.norm2.weight",
                             "decoder.decoder_blocks.2.norm2.bias",
                             "decoder.decoder_blocks.2.mlp.fc1.weight",
                             "decoder.decoder_blocks.2.mlp.fc1.bias",
                             "decoder.decoder_blocks.2.mlp.fc2.weight",
                             "decoder.decoder_blocks.2.mlp.fc2.bias",
                             "decoder.decoder_blocks.3.norm1.weight",
                             "decoder.decoder_blocks.3.norm1.bias",
                             "decoder.decoder_blocks.3.attn.qkv.weight",
                             "decoder.decoder_blocks.3.attn.qkv.bias",
                             "decoder.decoder_blocks.3.attn.proj.weight",
                             "decoder.decoder_blocks.3.attn.proj.bias",
                             "decoder.decoder_blocks.3.norm2.weight",
                             "decoder.decoder_blocks.3.norm2.bias",
                             "decoder.decoder_blocks.3.mlp.fc1.weight",
                             "decoder.decoder_blocks.3.mlp.fc1.bias",
                             "decoder.decoder_blocks.3.mlp.fc2.weight",
                             "decoder.decoder_blocks.3.mlp.fc2.bias",
                             "decoder.decoder_blocks.4.norm1.weight",
                             "decoder.decoder_blocks.4.norm1.bias",
                             "decoder.decoder_blocks.4.attn.qkv.weight",
                             "decoder.decoder_blocks.4.attn.qkv.bias",
                             "decoder.decoder_blocks.4.attn.proj.weight",
                             "decoder.decoder_blocks.4.attn.proj.bias",
                             "decoder.decoder_blocks.4.norm2.weight",
                             "decoder.decoder_blocks.4.norm2.bias",
                             "decoder.decoder_blocks.4.mlp.fc1.weight",
                             "decoder.decoder_blocks.4.mlp.fc1.bias",
                             "decoder.decoder_blocks.4.mlp.fc2.weight",
                             "decoder.decoder_blocks.4.mlp.fc2.bias",
                             "decoder.decoder_blocks.5.norm1.weight",
                             "decoder.decoder_blocks.5.norm1.bias",
                             "decoder.decoder_blocks.5.attn.qkv.weight",
                             "decoder.decoder_blocks.5.attn.qkv.bias",
                             "decoder.decoder_blocks.5.attn.proj.weight",
                             "decoder.decoder_blocks.5.attn.proj.bias",
                             "decoder.decoder_blocks.5.norm2.weight",
                             "decoder.decoder_blocks.5.norm2.bias",
                             "decoder.decoder_blocks.5.mlp.fc1.weight",
                             "decoder.decoder_blocks.5.mlp.fc1.bias",
                             "decoder.decoder_blocks.5.mlp.fc2.weight",
                             "decoder.decoder_blocks.5.mlp.fc2.bias",
                             "decoder.decoder_blocks.6.norm1.weight",
                             "decoder.decoder_blocks.6.norm1.bias",
                             "decoder.decoder_blocks.6.attn.qkv.weight",
                             "decoder.decoder_blocks.6.attn.qkv.bias",
                             "decoder.decoder_blocks.6.attn.proj.weight",
                             "decoder.decoder_blocks.6.attn.proj.bias",
                             "decoder.decoder_blocks.6.norm2.weight",
                             "decoder.decoder_blocks.6.norm2.bias",
                             "decoder.decoder_blocks.6.mlp.fc1.weight",
                             "decoder.decoder_blocks.6.mlp.fc1.bias",
                             "decoder.decoder_blocks.6.mlp.fc2.weight",
                             "decoder.decoder_blocks.6.mlp.fc2.bias",
                             "decoder.decoder_blocks.7.norm1.weight",
                             "decoder.decoder_blocks.7.norm1.bias",
                             "decoder.decoder_blocks.7.attn.qkv.weight",
                             "decoder.decoder_blocks.7.attn.qkv.bias",
                             "decoder.decoder_blocks.7.attn.proj.weight",
                             "decoder.decoder_blocks.7.attn.proj.bias",
                             "decoder.decoder_blocks.7.norm2.weight",
                             "decoder.decoder_blocks.7.norm2.bias",
                             "decoder.decoder_blocks.7.mlp.fc1.weight",
                             "decoder.decoder_blocks.7.mlp.fc1.bias",
                             "decoder.decoder_blocks.7.mlp.fc2.weight",
                             "decoder.decoder_blocks.7.mlp.fc2.bias",
                             "decoder.decoder_norm.weight",
                             "decoder.decoder_norm.bias",
                             "decoder.decoder_pred.weight",
                             "decoder.decoder_pred.bias",];
    let path = std::path::PathBuf::from("/Users/dalan/rustspace/ml-learning-demos/prithvi-model/Prithvi_EO_V2_300M_TL.safetensors");
    // let data = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path)? };
    // for name in weight_params.iter() {
    //     let data = data.load(name, &Device::Cpu)?;
    //     println!("key={:?}, params={}", name, data);
    // }

    let mut vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)? };
    let decoder_blocks = vb.pp("decoder").pp("mask_token");
    decoder_blocks.dtype();
    Ok(())
}

/// 基于gdal读取geotiff文件
fn process_geotiff_gdal<P: AsRef<std::path::Path>>(path: P, device: &Device) -> gdal::errors::Result<Tensor>{
    use gdal::Dataset;
    let dataset = Dataset::open(path)?;
    let rasterband = dataset.rasterband(1)?;
    let buffers = rasterband.read_band_as::<u8>()?.to_array()?;
    let datas = buffers.iter().map(|x| *x).collect::<Vec<u8>>();

    let data = Tensor::from_vec(datas, buffers.shape(), device).unwrap();

    Ok(data)
}
fn load_geotiff_as_tensor(path: &String, device: &Device) -> Result<Tensor>{
    let filename = std::path::PathBuf::from(path);
    let data = process_geotiff_gdal(&filename, device).unwrap();

    // 调整维度顺序为 [B, C, T, H, W]
    // let data = data.permute((1, 0, 2, 3))?; // 从 [B, C, H, W] 转为 [B, C, 1, H, W]
    // let data = data.expand((1, 6, 1, h, w))?; // 假设需要6个通道

    // 调整维度顺序为 [B, C, T, H, W]
    let data = data
        .permute((2, 0, 1))? // [c, h, w]
        .unsqueeze(0)?       // [1, c, h, w]
        .unsqueeze(2)?       // [1, c, 1, h, w]
        .expand((1, 6, 1, 448, 560))?; // 假设需要6个通道


    Ok(data)
}

fn load_tiff_as_tensor(path: &String, device: &Device) -> Result<Tensor>{
    let filename = std::path::PathBuf::from(path);
    let mut decoder = tiff::decoder::Decoder::new(File::open(filename)?).unwrap();
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
    // 假设 raw_data 是 Vec<i16>, 图像尺寸为 (w, h), 通道数 c
    let c = 1; // 根据实际通道数调整，例如 GeoTIFF 单波段则为 1
    let raw_data_f32: Vec<f32> = raw_data.into_iter().map(|x| x as f32).collect();

    // // 创建张量，假设数据布局为 (h, w, c)
    let data = Tensor::from_vec(raw_data_f32, (h, w, c), &Device::Cpu)?;
    // 调整维度顺序为 [B, C, T, H, W]
    // let data = data.permute((1, 0, 2, 3))?; // 从 [B, C, H, W] 转为 [B, C, 1, H, W]
    // let data = data.expand((1, 6, 1, h, w))?; // 假设需要6个通道

    // 调整维度顺序为 [B, C, T, H, W]
    let data = data
        .permute((2, 0, 1))? // [c, h, w]
        .unsqueeze(0)?       // [1, c, h, w]
        .unsqueeze(2)?       // [1, c, 1, h, w]
        .expand((1, 6, 1, h, w))?; // 假设需要6个通道


    Ok(data)
}