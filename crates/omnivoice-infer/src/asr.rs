use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::whisper::{
    self, model::Whisper as WhisperModel, quantized_model::Whisper as QuantizedWhisperModel, Config,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::error::{OmniVoiceError, Result};

const DEFAULT_LOCAL_ASR_MODEL: &str = "H:/omnivoice/model/whisper";
const DEFAULT_HF_ASR_MODEL: &str = "openai/whisper-large-v3-turbo";
const MEL_FILTERS_80: &[u8] = include_bytes!("../../../tools/whisper/melfilters.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("../../../tools/whisper/melfilters128.bytes");

#[derive(Debug)]
enum WhisperBackend {
    Standard(WhisperModel),
    Quantized(QuantizedWhisperModel),
}

impl WhisperBackend {
    fn encoder_forward(&mut self, mel: &Tensor, flush: bool) -> Result<Tensor> {
        match self {
            Self::Standard(model) => model.encoder.forward(mel, flush).map_err(Into::into),
            Self::Quantized(model) => model.encoder.forward(mel, flush).map_err(Into::into),
        }
    }

    fn decoder_forward(
        &mut self,
        tokens: &Tensor,
        audio_features: &Tensor,
        flush: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Standard(model) => model
                .decoder
                .forward(tokens, audio_features, flush)
                .map_err(Into::into),
            Self::Quantized(model) => model
                .decoder
                .forward(tokens, audio_features, flush)
                .map_err(Into::into),
        }
    }

    fn final_linear(&self, ys: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(model) => model.decoder.final_linear(ys).map_err(Into::into),
            Self::Quantized(model) => model.decoder.final_linear(ys).map_err(Into::into),
        }
    }

    fn config(&self) -> &Config {
        match self {
            Self::Standard(model) => &model.config,
            Self::Quantized(model) => &model.config,
        }
    }
}

#[derive(Debug)]
pub struct WhisperAsr {
    model: WhisperBackend,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_timestamps_token: u32,
    mel_filters: Vec<f32>,
}

impl WhisperAsr {
    pub fn load(model_id_or_path: &str, device: Device) -> Result<Self> {
        let requested = if model_id_or_path.is_empty() {
            DEFAULT_LOCAL_ASR_MODEL
        } else {
            model_id_or_path
        };
        let (config_path, tokenizer_path, weights_path, quantized) =
            resolve_model_files(requested)?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(OmniVoiceError::Tokenizer)?;
        let model = if quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_path,
                &device,
            )
            .map_err(OmniVoiceError::Candle)?;
            WhisperBackend::Quantized(
                QuantizedWhisperModel::load(&vb, config.clone()).map_err(OmniVoiceError::Candle)?,
            )
        } else {
            let vb = mmap_var_builder(&weights_path, whisper::DTYPE, &device)?;
            WhisperBackend::Standard(
                WhisperModel::load(&vb, config.clone()).map_err(OmniVoiceError::Candle)?,
            )
        };
        let no_timestamps_token = token_id(&tokenizer, whisper::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
            .map(|token| {
                if config.suppress_tokens.contains(&token) || token == no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        Ok(Self {
            model,
            tokenizer: tokenizer.clone(),
            config: config.clone(),
            suppress_tokens: Tensor::new(suppress_tokens.as_slice(), &device)?,
            sot_token: token_id(&tokenizer, whisper::SOT_TOKEN)?,
            transcribe_token: token_id(&tokenizer, whisper::TRANSCRIBE_TOKEN)?,
            eot_token: token_id(&tokenizer, whisper::EOT_TOKEN)?,
            no_timestamps_token,
            device,
            mel_filters: load_mel_filters(config.num_mel_bins)?,
        })
    }

    pub fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> Result<String> {
        let pcm = if sample_rate == whisper::SAMPLE_RATE as u32 {
            samples.to_vec()
        } else {
            crate::audio_input::resample_linear(samples, sample_rate, whisper::SAMPLE_RATE as u32)
        };
        let mel = whisper::audio::pcm_to_mel(&self.config, &pcm, &self.mel_filters);
        let mel_len = mel.len() / self.config.num_mel_bins;
        let mel = Tensor::from_vec(mel, (1, self.config.num_mel_bins, mel_len), &self.device)?;
        let audio_features = self.model.encoder_forward(&mel, true)?;
        let mut tokens = vec![
            self.sot_token,
            self.transcribe_token,
            self.no_timestamps_token,
        ];

        let sample_len = self.model.config().max_target_positions / 2;
        for index in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self
                .model
                .decoder_forward(&tokens_t, &audio_features, index == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?
                .broadcast_add(&self.suppress_tokens)?;
            let next_token = logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            if next_token == self.eot_token {
                break;
            }
            tokens.push(next_token);
        }

        self.tokenizer
            .decode(&tokens, true)
            .map(|text| text.trim().to_string())
            .map_err(Into::into)
    }
}

pub fn default_local_asr_model_path() -> &'static str {
    DEFAULT_LOCAL_ASR_MODEL
}

pub fn default_asr_model_spec() -> String {
    if Path::new(DEFAULT_LOCAL_ASR_MODEL).exists() {
        DEFAULT_LOCAL_ASR_MODEL.to_string()
    } else {
        DEFAULT_HF_ASR_MODEL.to_string()
    }
}

fn resolve_model_files(model_id_or_path: &str) -> Result<(PathBuf, PathBuf, PathBuf, bool)> {
    let local_path = Path::new(model_id_or_path);
    if local_path.exists() {
        let gguf = local_path
            .read_dir()?
            .filter_map(|entry| entry.ok().map(|item| item.path()))
            .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("gguf"));
        if let Some(weights) = gguf {
            return Ok((
                local_path.join("config.json"),
                local_path.join("tokenizer.json"),
                weights,
                true,
            ));
        }
        return Ok((
            local_path.join("config.json"),
            local_path.join("tokenizer.json"),
            local_path.join("model.safetensors"),
            false,
        ));
    }
    let api = Api::new().map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    let repo = api.repo(Repo::with_revision(
        model_id_or_path.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let config = repo
        .get("config.json")
        .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    let tokenizer = repo
        .get("tokenizer.json")
        .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    let weights = repo
        .get("model.safetensors")
        .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    Ok((config, tokenizer, weights, false))
}

fn load_mel_filters(num_mel_bins: usize) -> Result<Vec<f32>> {
    let bytes = match num_mel_bins {
        80 => MEL_FILTERS_80,
        128 => MEL_FILTERS_128,
        other => {
            return Err(OmniVoiceError::Unsupported(format!(
                "unsupported Whisper mel bin count {other}"
            )))
        }
    };
    let mut filters = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        filters.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(filters)
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        OmniVoiceError::InvalidData(format!("Whisper tokenizer is missing token {token}"))
    })
}

fn mmap_var_builder(
    weights_path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    let paths = [weights_path];
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? })
}
