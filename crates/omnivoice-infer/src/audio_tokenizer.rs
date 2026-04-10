use std::{fs, path::Path, sync::OnceLock};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use serde::Deserialize;

use crate::{
    artifacts::{AudioTokenizerArtifacts, RuntimeArtifacts},
    error::{OmniVoiceError, Result},
    runtime::RuntimeOptions,
};

#[path = "audio_tokenizer_dac.rs"]
mod audio_tokenizer_dac;
#[path = "audio_tokenizer_hubert.rs"]
mod audio_tokenizer_hubert;

use audio_tokenizer_dac::{AcousticEncoder, ResidualVectorQuantizer};
use audio_tokenizer_hubert::{HubertModel, SemanticEncoder};

#[derive(Debug, Clone, Deserialize)]
struct AudioTokenizerConfigFile {
    target_bandwidths: Vec<f32>,
    sample_rate: u32,
    semantic_sample_rate: u32,
    downsample_factor: usize,
    codebook_size: usize,
    codebook_dim: usize,
    kernel_size: usize,
    channel_ratios: Vec<usize>,
    strides: Vec<usize>,
    block_dilations: Vec<usize>,
    unit_kernel_size: usize,
    acoustic_model_config: AcousticModelConfigFile,
    semantic_model_config: SemanticModelConfigFile,
}

#[derive(Debug, Clone, Deserialize)]
struct AcousticModelConfigFile {
    encoder_hidden_size: usize,
    downsampling_ratios: Vec<usize>,
    hidden_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct SemanticModelConfigFile {
    conv_bias: bool,
    conv_dim: Vec<usize>,
    conv_kernel: Vec<usize>,
    conv_stride: Vec<usize>,
    feat_extract_activation: String,
    feat_extract_norm: String,
    feat_proj_layer_norm: bool,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    num_attention_heads: usize,
    num_conv_pos_embedding_groups: usize,
    num_conv_pos_embeddings: usize,
    num_hidden_layers: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct AudioTokenizerModelConfig {
    pub sample_rate: u32,
    pub semantic_sample_rate: u32,
    pub downsample_factor: usize,
    pub target_bandwidths: Vec<f32>,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub kernel_size: usize,
    pub channel_ratios: Vec<usize>,
    pub strides: Vec<usize>,
    pub block_dilations: Vec<usize>,
    pub unit_kernel_size: usize,
    pub acoustic_hidden_size: usize,
    pub acoustic_encoder_hidden_size: usize,
    pub acoustic_downsampling_ratios: Vec<usize>,
    pub semantic_hidden_size: usize,
    pub semantic_intermediate_size: usize,
    pub semantic_num_heads: usize,
    pub semantic_num_layers: usize,
    pub semantic_layer_norm_eps: f64,
    pub semantic_conv_bias: bool,
    pub semantic_conv_dim: Vec<usize>,
    pub semantic_conv_kernel: Vec<usize>,
    pub semantic_conv_stride: Vec<usize>,
    pub semantic_feat_extract_norm: String,
    pub semantic_feat_extract_activation: candle_nn::Activation,
    pub semantic_feat_proj_layer_norm: bool,
    pub semantic_hidden_activation: candle_nn::Activation,
    pub semantic_num_conv_pos_embeddings: usize,
    pub semantic_num_conv_pos_groups: usize,
}

impl AudioTokenizerModelConfig {
    pub fn from_artifacts(audio_tokenizer: &AudioTokenizerArtifacts) -> Result<Self> {
        let raw: AudioTokenizerConfigFile =
            serde_json::from_str(&fs::read_to_string(audio_tokenizer.config_path())?)?;
        Ok(Self {
            sample_rate: raw.sample_rate,
            semantic_sample_rate: raw.semantic_sample_rate,
            downsample_factor: raw.downsample_factor,
            target_bandwidths: raw.target_bandwidths,
            codebook_size: raw.codebook_size,
            codebook_dim: raw.codebook_dim,
            kernel_size: raw.kernel_size,
            channel_ratios: raw.channel_ratios,
            strides: raw.strides,
            block_dilations: raw.block_dilations,
            unit_kernel_size: raw.unit_kernel_size,
            acoustic_hidden_size: raw.acoustic_model_config.hidden_size,
            acoustic_encoder_hidden_size: raw.acoustic_model_config.encoder_hidden_size,
            acoustic_downsampling_ratios: raw.acoustic_model_config.downsampling_ratios,
            semantic_hidden_size: raw.semantic_model_config.hidden_size,
            semantic_intermediate_size: raw.semantic_model_config.intermediate_size,
            semantic_num_heads: raw.semantic_model_config.num_attention_heads,
            semantic_num_layers: raw.semantic_model_config.num_hidden_layers,
            semantic_layer_norm_eps: raw.semantic_model_config.layer_norm_eps,
            semantic_conv_bias: raw.semantic_model_config.conv_bias,
            semantic_conv_dim: raw.semantic_model_config.conv_dim,
            semantic_conv_kernel: raw.semantic_model_config.conv_kernel,
            semantic_conv_stride: raw.semantic_model_config.conv_stride,
            semantic_feat_extract_norm: raw.semantic_model_config.feat_extract_norm,
            semantic_feat_extract_activation: parse_activation(
                &raw.semantic_model_config.feat_extract_activation,
            )?,
            semantic_feat_proj_layer_norm: raw.semantic_model_config.feat_proj_layer_norm,
            semantic_hidden_activation: parse_activation(&raw.semantic_model_config.hidden_act)?,
            semantic_num_conv_pos_embeddings: raw.semantic_model_config.num_conv_pos_embeddings,
            semantic_num_conv_pos_groups: raw.semantic_model_config.num_conv_pos_embedding_groups,
        })
    }

    pub fn hop_length(&self) -> usize {
        self.acoustic_downsampling_ratios.iter().product()
    }

    pub fn num_quantizers(&self) -> usize {
        let bandwidth = self.target_bandwidths.last().copied().unwrap_or(2.0);
        let frame_rate = (self.sample_rate as f32 / self.hop_length() as f32).ceil();
        let codebook_bits = (self.codebook_size as f32).log2();
        ((1000.0 * bandwidth) / (frame_rate * codebook_bits))
            .floor()
            .max(1.0) as usize
    }

    pub fn semantic_downsample_factor(&self) -> usize {
        ((self.hop_length() as f32 / (self.sample_rate as f32 / self.semantic_sample_rate as f32))
            / self.downsample_factor as f32)
            .round()
            .max(1.0) as usize
    }
}

#[derive(Debug)]
pub struct AudioTokenizerRuntimePlan {
    device: Device,
    runtime_dtype: DType,
    config: AudioTokenizerModelConfig,
    weights_path: std::path::PathBuf,
    model: OnceLock<std::result::Result<AudioTokenizerModel, String>>,
}

impl AudioTokenizerRuntimePlan {
    pub fn from_runtime_artifacts(
        options: RuntimeOptions,
        runtime: &RuntimeArtifacts,
    ) -> Result<Self> {
        let device = options.resolve_device()?;
        let runtime_dtype = options.resolve_dtype_for_runtime_device(&device);
        Ok(Self {
            device,
            runtime_dtype,
            config: AudioTokenizerModelConfig::from_artifacts(runtime.audio_tokenizer())?,
            weights_path: runtime.audio_tokenizer().weights_path().to_path_buf(),
            model: OnceLock::new(),
        })
    }

    pub fn encode_waveform(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<crate::contracts::I64Tensor2> {
        let codes = self.encode_waveform_device(samples, sample_rate)?;
        let (quantizers, steps) = codes.dims2()?;
        let data = codes
            .to_device(&Device::Cpu)?
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        crate::contracts::I64Tensor2::new((quantizers, steps), data)
    }

    pub fn encode_waveform_device(&self, samples: &[f32], sample_rate: u32) -> Result<Tensor> {
        if sample_rate != self.config.sample_rate {
            return Err(OmniVoiceError::InvalidRequest(format!(
                "audio tokenizer expects {} Hz input, got {sample_rate}",
                self.config.sample_rate
            )));
        }
        let waveform = Tensor::from_vec(samples.to_vec(), (1, 1, samples.len()), &self.device)?;
        let semantic_waveform = if self.config.sample_rate != self.config.semantic_sample_rate {
            let semantic_samples = crate::audio_input::resample_linear(
                samples,
                self.config.sample_rate,
                self.config.semantic_sample_rate,
            );
            let semantic_len = semantic_samples.len();
            Some(Tensor::from_vec(
                semantic_samples,
                (1, semantic_len),
                &self.device,
            )?)
        } else {
            None
        };
        let codes = self
            .model()?
            .encode(&waveform, semantic_waveform.as_ref())?;
        Ok(codes.i(0)?.to_dtype(DType::I64)?)
    }

    fn model(&self) -> Result<&AudioTokenizerModel> {
        let result = self.model.get_or_init(|| {
            AudioTokenizerModel::load(
                &self.config,
                &self.weights_path,
                &self.device,
                self.runtime_dtype,
            )
            .map_err(|error| error.to_string())
        });
        match result {
            Ok(model) => Ok(model),
            Err(message) => Err(OmniVoiceError::InvalidData(message.clone())),
        }
    }
}

#[derive(Debug)]
struct AudioTokenizerModel {
    semantic_model: HubertModel,
    encoder_semantic: SemanticEncoder,
    acoustic_encoder: AcousticEncoder,
    fc: Linear,
    quantizer: ResidualVectorQuantizer,
    config: AudioTokenizerModelConfig,
}

impl AudioTokenizerModel {
    fn load(
        config: &AudioTokenizerModelConfig,
        weights_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = mmap_var_builder(weights_path, dtype, device)?;
        Ok(Self {
            semantic_model: HubertModel::load(config, vb.pp("semantic_model"))?,
            encoder_semantic: SemanticEncoder::load(config, vb.pp("encoder_semantic"))?,
            acoustic_encoder: AcousticEncoder::load(config, vb.pp("acoustic_encoder"))?,
            fc: load_linear(
                vb.pp("fc"),
                config.acoustic_hidden_size + config.semantic_hidden_size,
                config.acoustic_hidden_size + config.semantic_hidden_size,
                true,
            )?,
            quantizer: ResidualVectorQuantizer::load(config, vb.pp("quantizer"))?,
            config: config.clone(),
        })
    }

    fn encode(&self, waveform: &Tensor, semantic_waveform: Option<&Tensor>) -> Result<Tensor> {
        let (_, channels, _) = waveform.dims3()?;
        if channels != 1 {
            return Err(OmniVoiceError::InvalidTensorShape {
                name: "audio_tokenizer.input_values".to_string(),
                expected: "(B, 1, T)".to_string(),
                actual: format!("{:?}", waveform.dims()),
            });
        }
        let semantic_input = match semantic_waveform {
            Some(semantic_waveform) => semantic_waveform.clone(),
            None => waveform.i((.., 0, ..))?,
        };
        let semantic_features = self
            .semantic_model
            .extract_semantic_features_from_resampled(
                &semantic_input,
                self.config.semantic_downsample_factor(),
            )?;
        let semantic_latents = self
            .encoder_semantic
            .forward(&semantic_features.transpose(1, 2)?)?;

        let acoustic_latents = {
            let raw = self.acoustic_encoder.forward(waveform)?;
            if raw.dim(candle_core::D::Minus1)? == semantic_latents.dim(candle_core::D::Minus1)? {
                raw
            } else {
                self.acoustic_encoder.forward(&waveform.pad_with_zeros(
                    candle_core::D::Minus1,
                    self.config.hop_length() / 2,
                    self.config.hop_length() / 2,
                )?)?
            }
        };

        let embeddings = Tensor::cat(&[&acoustic_latents, &semantic_latents], 1)?;
        let embeddings = self
            .fc
            .forward(&embeddings.transpose(1, 2)?)?
            .transpose(1, 2)?;
        self.quantizer.encode(&embeddings)
    }
}

pub(crate) fn load_linear(
    vb: VarBuilder<'_>,
    in_dim: usize,
    out_dim: usize,
    with_bias: bool,
) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = if with_bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

pub(crate) fn parse_activation(name: &str) -> Result<candle_nn::Activation> {
    match name {
        "gelu" => Ok(candle_nn::Activation::Gelu),
        "silu" => Ok(candle_nn::Activation::Silu),
        "elu" => Ok(candle_nn::Activation::Elu(1.0)),
        other => Err(OmniVoiceError::Unsupported(format!(
            "unsupported activation {other}"
        ))),
    }
}

fn mmap_var_builder(
    weights_path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let paths = [weights_path];
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? })
}
