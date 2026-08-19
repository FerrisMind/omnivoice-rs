//! GGUF-friendly codec weights: keep Linear/Embedding packed; dequant Conv only per forward.

use std::{
    path::Path,
    sync::Arc,
};

use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Module, Result as CandleResult, Tensor,
};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Linear, VarBuilder,
};
use candle_transformers::quantized_var_builder::VarBuilder as QuantizedVarBuilder;

use crate::error::Result;

/// Lookup table that stays in GGUF storage and gathers rows on demand.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    weight: Arc<QTensor>,
    activation_dtype: DType,
    device: Device,
}

impl QuantizedEmbedding {
    pub fn new(weight: Arc<QTensor>, activation_dtype: DType, device: Device) -> Self {
        Self {
            weight,
            activation_dtype,
            device,
        }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.weight
            .embedding(input_ids)?
            .to_dtype(self.activation_dtype)
            .map_err(Into::into)
    }

    /// Full codebook materialization for VQ distance search (encode path only).
    pub fn embeddings(&self) -> Result<Tensor> {
        self.weight
            .dequantize(&self.device)?
            .to_dtype(self.activation_dtype)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct CodecEmbedding {
    inner: CodecEmbeddingInner,
}

#[derive(Debug, Clone)]
enum CodecEmbeddingInner {
    Dense(Embedding),
    Quantized(QuantizedEmbedding),
}

impl CodecEmbedding {
    pub fn embeddings(&self) -> Result<Tensor> {
        match &self.inner {
            CodecEmbeddingInner::Dense(layer) => Ok(layer.embeddings().clone()),
            CodecEmbeddingInner::Quantized(layer) => layer.embeddings(),
        }
    }
}

impl Module for CodecEmbedding {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecEmbeddingInner::Dense(layer) => layer.forward(xs),
            CodecEmbeddingInner::Quantized(layer) => {
                layer.forward(xs).map_err(codec_candle_err)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodecLinear {
    inner: CodecLinearInner,
}

#[derive(Debug, Clone)]
enum CodecLinearInner {
    Dense(Linear),
    Quantized {
        weight: QMatMul,
        bias: Option<Tensor>,
    },
}

impl Module for CodecLinear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecLinearInner::Dense(layer) => layer.forward(xs),
            CodecLinearInner::Quantized { weight, bias } => {
                let xs = xs.contiguous()?;
                let mut out = xs.apply(weight)?;
                if let Some(bias) = bias {
                    out = out.broadcast_add(bias)?;
                }
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodecConv1d {
    inner: CodecConv1dInner,
}

#[derive(Debug, Clone)]
enum CodecConv1dInner {
    Dense(Conv1d),
    Lazy(LazyConv1dWeights),
    WeightNormLazy(LazyWeightNormConv1d),
}

#[derive(Debug, Clone)]
struct LazyConv1dWeights {
    weight: Arc<QTensor>,
    bias: Option<Tensor>,
    config: Conv1dConfig,
    activation_dtype: DType,
    device: Device,
}

#[derive(Debug, Clone)]
struct LazyWeightNormConv1d {
    g: Arc<QTensor>,
    v: Arc<QTensor>,
    bias: Option<Tensor>,
    config: Conv1dConfig,
    activation_dtype: DType,
    device: Device,
}

impl Module for CodecConv1d {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecConv1dInner::Dense(layer) => layer.forward(xs),
            CodecConv1dInner::Lazy(lazy) => lazy.forward(xs).map_err(codec_candle_err),
            CodecConv1dInner::WeightNormLazy(lazy) => lazy.forward(xs).map_err(codec_candle_err),
        }
    }
}

impl LazyConv1dWeights {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let weight = self
            .weight
            .dequantize(&self.device)?
            .to_dtype(self.activation_dtype)?;
        Conv1d::new(weight, self.bias.clone(), self.config)
            .forward(xs)
            .map_err(Into::into)
    }
}

impl LazyWeightNormConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = self.activation_dtype;
        let g = self.g.dequantize(&self.device)?.to_dtype(dtype)?;
        let v = self.v.dequantize(&self.device)?.to_dtype(dtype)?;
        let norm = v.sqr()?.sum_keepdim((0, 1))?.sqrt()?;
        let scale = g.broadcast_div(&norm)?;
        let weight = v.broadcast_mul(&scale.broadcast_as(v.shape().dims())?)?;
        Conv1d::new(weight, self.bias.clone(), self.config)
            .forward(xs)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct CodecConvTranspose1d {
    inner: CodecConvTranspose1dInner,
}

#[derive(Debug, Clone)]
enum CodecConvTranspose1dInner {
    Dense(ConvTranspose1d),
    Lazy(LazyConvTranspose1dWeights),
}

#[derive(Debug, Clone)]
struct LazyConvTranspose1dWeights {
    weight: Arc<QTensor>,
    bias: Option<Tensor>,
    config: ConvTranspose1dConfig,
    activation_dtype: DType,
    device: Device,
}

impl Module for CodecConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecConvTranspose1dInner::Dense(layer) => layer.forward(xs),
            CodecConvTranspose1dInner::Lazy(lazy) => lazy.forward(xs).map_err(codec_candle_err),
        }
    }
}

impl LazyConvTranspose1dWeights {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let weight = self
            .weight
            .dequantize(&self.device)?
            .to_dtype(self.activation_dtype)?;
        ConvTranspose1d::new(weight, self.bias.clone(), self.config)
            .forward(xs)
            .map_err(Into::into)
    }
}

fn codec_candle_err(error: crate::error::OmniVoiceError) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}

#[derive(Clone)]
pub struct CodecWeightSource {
    inner: CodecWeightSourceInner,
}

#[derive(Clone)]
enum CodecWeightSourceInner {
    Dense(VarBuilder<'static>),
    Quantized {
        qvb: QuantizedVarBuilder,
        activation_dtype: DType,
        device: Device,
    },
}

impl CodecWeightSource {
    pub fn from_weight_path(path: &Path, activation_dtype: DType, device: &Device) -> Result<Self> {
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            let qvb = QuantizedVarBuilder::from_gguf(path, device)?;
            Ok(Self {
                inner: CodecWeightSourceInner::Quantized {
                    qvb,
                    activation_dtype,
                    device: device.clone(),
                },
            })
        } else {
            let paths = [path];
            // SAFETY: read-only mmap of immutable weight files for process lifetime.
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, activation_dtype, device)? };
            Ok(Self {
                inner: CodecWeightSourceInner::Dense(vb),
            })
        }
    }

    pub fn pp(&self, suffix: impl ToString) -> Self {
        let suffix = suffix.to_string();
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => Self {
                inner: CodecWeightSourceInner::Dense(vb.pp(suffix.clone())),
            },
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => Self {
                inner: CodecWeightSourceInner::Quantized {
                    qvb: qvb.pp(suffix),
                    activation_dtype: *activation_dtype,
                    device: device.clone(),
                },
            },
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.inner, CodecWeightSourceInner::Quantized { .. })
    }

    pub fn get(&self, shape: impl Into<candle_core::Shape>, name: &str) -> Result<Tensor> {
        let shape = shape.into();
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => vb.get(shape, name).map_err(Into::into),
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => qvb
                .get(shape, name)?
                .dequantize(device)?
                .to_dtype(*activation_dtype)
                .map_err(Into::into),
        }
    }

    pub fn load_linear(
        &self,
        in_dim: usize,
        out_dim: usize,
        with_bias: bool,
    ) -> Result<CodecLinear> {
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = vb.get((out_dim, in_dim), "weight")?;
                let bias = if with_bias {
                    Some(vb.get(out_dim, "bias")?)
                } else {
                    None
                };
                Ok(CodecLinear {
                    inner: CodecLinearInner::Dense(Linear::new(weight, bias)),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                let weight = keep_quantized_matmul(qvb.get((out_dim, in_dim), "weight")?)?;
                let bias = if with_bias {
                    Some(
                        qvb.get(out_dim, "bias")?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    )
                } else {
                    None
                };
                Ok(CodecLinear {
                    inner: CodecLinearInner::Quantized { weight, bias },
                })
            }
        }
    }

    pub fn load_embedding(
        &self,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<CodecEmbedding> {
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => Ok(CodecEmbedding {
                inner: CodecEmbeddingInner::Dense(Embedding::new(
                    vb.get((vocab_size, hidden_size), "embed")?,
                    hidden_size,
                )),
            }),
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => Ok(CodecEmbedding {
                inner: CodecEmbeddingInner::Quantized(QuantizedEmbedding::new(
                    qvb.get((vocab_size, hidden_size), "embed")?,
                    *activation_dtype,
                    device.clone(),
                )),
            }),
        }
    }

    pub fn load_embedding_at(
        &self,
        vocab_size: usize,
        hidden_size: usize,
        name: &str,
    ) -> Result<CodecEmbedding> {
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => Ok(CodecEmbedding {
                inner: CodecEmbeddingInner::Dense(Embedding::new(
                    vb.get((vocab_size, hidden_size), name)?,
                    hidden_size,
                )),
            }),
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => Ok(CodecEmbedding {
                inner: CodecEmbeddingInner::Quantized(QuantizedEmbedding::new(
                    qvb.get((vocab_size, hidden_size), name)?,
                    *activation_dtype,
                    device.clone(),
                )),
            }),
        }
    }

    pub fn load_linear_named(
        &self,
        in_dim: usize,
        out_dim: usize,
        with_bias: bool,
        weight_name: &str,
        bias_name: &str,
    ) -> Result<CodecLinear> {
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = vb.get((out_dim, in_dim), weight_name)?;
                let bias = if with_bias {
                    Some(vb.get(out_dim, bias_name)?)
                } else {
                    None
                };
                Ok(CodecLinear {
                    inner: CodecLinearInner::Dense(Linear::new(weight, bias)),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                let weight = keep_quantized_matmul(qvb.get((out_dim, in_dim), weight_name)?)?;
                let bias = if with_bias {
                    Some(
                        qvb.get(out_dim, bias_name)?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    )
                } else {
                    None
                };
                Ok(CodecLinear {
                    inner: CodecLinearInner::Quantized { weight, bias },
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_conv1d(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        with_bias: bool,
        groups: usize,
    ) -> Result<CodecConv1d> {
        let config = Conv1dConfig {
            stride,
            padding,
            dilation,
            groups,
            ..Default::default()
        };
        let in_per_group = in_channels / groups;
        let weight_shape = (out_channels, in_per_group, kernel_size);
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = vb.get(weight_shape, "weight")?;
                let bias = if with_bias {
                    Some(vb.get(out_channels, "bias")?)
                } else {
                    None
                };
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::Dense(Conv1d::new(weight, bias, config)),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                let weight = qvb.get(weight_shape, "weight")?;
                let bias = if with_bias {
                    Some(
                        qvb.get(out_channels, "bias")?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    )
                } else {
                    None
                };
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::Lazy(LazyConv1dWeights {
                        weight,
                        bias,
                        config,
                        activation_dtype: *activation_dtype,
                        device: device.clone(),
                    }),
                })
            }
        }
    }

    pub fn load_conv1d_flexible(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        padding: usize,
    ) -> Result<CodecConv1d> {
        let config = Conv1dConfig {
            dilation,
            padding,
            ..Default::default()
        };
        let expected_in = in_channels / config.groups;
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = match vb.get((out_channels, expected_in, kernel_size), "weight") {
                    Ok(weight) => weight,
                    Err(_) if out_channels == 1 => vb
                        .get((expected_in, kernel_size), "weight")?
                        .reshape((out_channels, expected_in, kernel_size))?,
                    Err(error) => return Err(error.into()),
                };
                let bias = vb.get(out_channels, "bias")?;
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::Dense(Conv1d::new(weight, Some(bias), config)),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                if let Ok(weight) = qvb.get((out_channels, expected_in, kernel_size), "weight") {
                    let bias = Some(
                        qvb.get(out_channels, "bias")?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    );
                    return Ok(CodecConv1d {
                        inner: CodecConv1dInner::Lazy(LazyConv1dWeights {
                            weight,
                            bias,
                            config,
                            activation_dtype: *activation_dtype,
                            device: device.clone(),
                        }),
                    });
                }
                if out_channels == 1 {
                    let weight = qvb
                        .get((expected_in, kernel_size), "weight")?
                        .dequantize(device)?
                        .to_dtype(*activation_dtype)?
                        .reshape((out_channels, expected_in, kernel_size))?;
                    let bias = vb_get_bias(qvb, out_channels, device, *activation_dtype)?;
                    return Ok(CodecConv1d {
                        inner: CodecConv1dInner::Dense(Conv1d::new(weight, bias, config)),
                    });
                }
                Err(candle_core::Error::msg("missing conv1d weight").into())
            }
        }
    }

    pub fn load_weight_norm_conv1d(
        &self,
        out_channels: usize,
        in_channels_per_group: usize,
        kernel_size: usize,
        padding: usize,
        groups: usize,
    ) -> Result<CodecConv1d> {
        let config = Conv1dConfig {
            padding,
            groups,
            ..Default::default()
        };
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                if let Ok(weight) = vb.get((out_channels, in_channels_per_group, kernel_size), "weight")
                {
                    return Ok(CodecConv1d {
                        inner: CodecConv1dInner::Dense(Conv1d::new(
                            weight,
                            Some(vb.get(out_channels, "bias")?),
                            config,
                        )),
                    });
                }
                let g = vb.get((1, 1, kernel_size), "parametrizations.weight.original0")?;
                let v = vb.get(
                    (out_channels, in_channels_per_group, kernel_size),
                    "parametrizations.weight.original1",
                )?;
                let norm = v.sqr()?.sum_keepdim((0, 1))?.sqrt()?;
                let scale = g.broadcast_div(&norm)?;
                let weight = v.broadcast_mul(&scale.broadcast_as(v.shape().dims())?)?;
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::Dense(Conv1d::new(
                        weight,
                        Some(vb.get(out_channels, "bias")?),
                        config,
                    )),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                if let Ok(weight) =
                    qvb.get((out_channels, in_channels_per_group, kernel_size), "weight")
                {
                    let bias = Some(
                        qvb.get(out_channels, "bias")?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    );
                    return Ok(CodecConv1d {
                        inner: CodecConv1dInner::Lazy(LazyConv1dWeights {
                            weight,
                            bias,
                            config,
                            activation_dtype: *activation_dtype,
                            device: device.clone(),
                        }),
                    });
                }
                let g = qvb.get((1, 1, kernel_size), "parametrizations.weight.original0")?;
                let v = qvb.get(
                    (out_channels, in_channels_per_group, kernel_size),
                    "parametrizations.weight.original1",
                )?;
                let bias = Some(
                    qvb.get(out_channels, "bias")?
                        .dequantize(device)?
                        .to_dtype(*activation_dtype)?,
                );
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::WeightNormLazy(LazyWeightNormConv1d {
                        g,
                        v,
                        bias,
                        config,
                        activation_dtype: *activation_dtype,
                        device: device.clone(),
                    }),
                })
            }
        }
    }

    pub fn load_conv_transpose1d(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<CodecConvTranspose1d> {
        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding,
            ..Default::default()
        };
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = vb.get(
                    (in_channels, out_channels / config.groups, kernel_size),
                    "weight",
                )?;
                let bias = vb.get(out_channels, "bias")?;
                Ok(CodecConvTranspose1d {
                    inner: CodecConvTranspose1dInner::Dense(ConvTranspose1d::new(
                        weight,
                        Some(bias),
                        config,
                    )),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                let weight = qvb.get(
                    (in_channels, out_channels / config.groups, kernel_size),
                    "weight",
                )?;
                let bias = Some(
                    qvb.get(out_channels, "bias")?
                        .dequantize(device)?
                        .to_dtype(*activation_dtype)?,
                );
                Ok(CodecConvTranspose1d {
                    inner: CodecConvTranspose1dInner::Lazy(LazyConvTranspose1dWeights {
                        weight,
                        bias,
                        config,
                        activation_dtype: *activation_dtype,
                        device: device.clone(),
                    }),
                })
            }
        }
    }
}

fn keep_quantized_matmul(weight: Arc<QTensor>) -> Result<QMatMul> {
    let dtype = weight.dtype();
    match dtype {
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
            let device = weight.device();
            let w = weight.dequantize_f16(&device)?;
            Ok(QMatMul::TensorF16(w))
        }
        _ => Ok(QMatMul::QTensor(weight)),
    }
}

fn vb_get_bias(
    qvb: &QuantizedVarBuilder,
    out_channels: usize,
    device: &Device,
    activation_dtype: DType,
) -> Result<Option<Tensor>> {
    Ok(Some(
        qvb.get(out_channels, "bias")?
            .dequantize(device)?
            .to_dtype(activation_dtype)?,
    ))
}
