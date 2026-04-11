use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::whisper::{
    self, model::Whisper as WhisperModel, quantized_model::Whisper as QuantizedWhisperModel, Config,
};
use hf_hub::{
    api::{sync::Api, Siblings},
    Repo, RepoType,
};
use tokenizers::Tokenizer;

use crate::{
    error::{OmniVoiceError, Result},
    model_source::DEFAULT_WHISPER_REPO,
};

const DEFAULT_LOCAL_ASR_DIR_NAME: &str = "whisper";
const DEFAULT_HF_ASR_MODEL: &str = DEFAULT_WHISPER_REPO;
const DEFAULT_WHISPER_CONFIG_FILE: &str = "config.json";
const DEFAULT_WHISPER_TOKENIZER_FILE: &str = "tokenizer.json";
const DEFAULT_WHISPER_Q4_0_FILE: &str = "whisper-large-v3-turbo-q4_0.gguf";
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
        let requested = if model_id_or_path.trim().is_empty() {
            DEFAULT_HF_ASR_MODEL.to_string()
        } else {
            model_id_or_path.trim().to_string()
        };
        let (config_path, tokenizer_path, weights_path, quantized) =
            resolve_model_files(&requested)?;
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
    DEFAULT_LOCAL_ASR_DIR_NAME
}

pub fn default_asr_model_spec(model_root: Option<&Path>) -> String {
    if let Some(model_root) = model_root {
        let local_path = model_root.join(DEFAULT_LOCAL_ASR_DIR_NAME);
        if local_path.exists() {
            return local_path.display().to_string();
        }
    }
    DEFAULT_HF_ASR_MODEL.to_string()
}

fn resolve_model_files(model_id_or_path: &str) -> Result<(PathBuf, PathBuf, PathBuf, bool)> {
    let local_path = Path::new(model_id_or_path);
    if local_path.exists() {
        if let Some(weights) = find_local_whisper_weights(local_path)? {
            return Ok((
                local_path.join(DEFAULT_WHISPER_CONFIG_FILE),
                local_path.join(DEFAULT_WHISPER_TOKENIZER_FILE),
                weights,
                true,
            ));
        }
        return Ok((
            local_path.join(DEFAULT_WHISPER_CONFIG_FILE),
            local_path.join(DEFAULT_WHISPER_TOKENIZER_FILE),
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
    let repo_info = repo
        .info()
        .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    let remote_assets = find_remote_whisper_assets(model_id_or_path, &repo_info.siblings)?;
    let config = repo
        .get(&remote_assets.config)
        .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    let tokenizer = repo
        .get(&remote_assets.tokenizer)
        .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    let weights = repo
        .get(&remote_assets.weights)
        .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    Ok((config, tokenizer, weights, true))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperRemoteAssets {
    config: String,
    tokenizer: String,
    weights: String,
}

fn find_remote_whisper_assets(
    model_id_or_path: &str,
    siblings: &[Siblings],
) -> Result<WhisperRemoteAssets> {
    if model_id_or_path == DEFAULT_HF_ASR_MODEL {
        return Ok(WhisperRemoteAssets {
            config: find_exact_remote_file(siblings, DEFAULT_WHISPER_CONFIG_FILE)?,
            tokenizer: find_exact_remote_file(siblings, DEFAULT_WHISPER_TOKENIZER_FILE)?,
            weights: find_exact_remote_file(siblings, DEFAULT_WHISPER_Q4_0_FILE)?,
        });
    }

    Ok(WhisperRemoteAssets {
        config: find_whisper_repo_file(siblings, |path| {
            path.ends_with(DEFAULT_WHISPER_CONFIG_FILE)
        })?,
        tokenizer: find_whisper_repo_file(siblings, |path| {
            path.ends_with(DEFAULT_WHISPER_TOKENIZER_FILE)
        })?,
        weights: find_remote_whisper_weights(siblings)?,
    })
}

fn find_exact_remote_file(siblings: &[Siblings], file_name: &str) -> Result<String> {
    siblings
        .iter()
        .map(|sibling| sibling.rfilename.as_str())
        .find(|path| path == &file_name)
        .map(str::to_string)
        .ok_or_else(|| {
            OmniVoiceError::InvalidData(format!(
                "remote Whisper repo is missing required Candle file `{file_name}`"
            ))
        })
}

fn find_whisper_repo_file(
    siblings: &[Siblings],
    predicate: impl Fn(&str) -> bool,
) -> Result<String> {
    siblings
        .iter()
        .map(|sibling| sibling.rfilename.as_str())
        .find(|path| predicate(path))
        .map(str::to_string)
        .ok_or_else(|| {
            OmniVoiceError::InvalidData(
                "remote Whisper repo is missing the required q4_0/config/tokenizer files"
                    .to_string(),
            )
        })
}

fn find_local_whisper_weights(local_path: &Path) -> Result<Option<PathBuf>> {
    let mut candidates = local_path
        .read_dir()?
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter(|path| is_supported_whisper_gguf(path.to_string_lossy().as_ref()))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| compare_whisper_weight_paths(left.as_path(), right.as_path()));
    Ok(candidates.into_iter().next())
}

fn find_remote_whisper_weights(siblings: &[Siblings]) -> Result<String> {
    let mut candidates = siblings
        .iter()
        .map(|sibling| sibling.rfilename.as_str())
        .filter(|path| is_supported_whisper_gguf(path))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| compare_whisper_weight_names(left, right));
    candidates
        .into_iter()
        .next()
        .map(str::to_string)
        .ok_or_else(|| {
            OmniVoiceError::InvalidData(
                "remote Whisper repo is missing the required gguf/config/tokenizer files"
                    .to_string(),
            )
        })
}

fn is_supported_whisper_gguf(path: &str) -> bool {
    let normalized = path.to_ascii_lowercase().replace('\\', "/");
    normalized.ends_with(".gguf") && !normalized.contains("whisper.cpp/")
}

fn compare_whisper_weight_paths(left: &Path, right: &Path) -> std::cmp::Ordering {
    compare_whisper_weight_names(
        left.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default(),
        right
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default(),
    )
}

fn compare_whisper_weight_names(left: &str, right: &str) -> std::cmp::Ordering {
    whisper_weight_rank(left)
        .cmp(&whisper_weight_rank(right))
        .then_with(|| left.cmp(right))
}

fn whisper_weight_rank(path: &str) -> usize {
    let normalized = path.to_ascii_lowercase();
    if normalized.contains("q4_0") {
        0
    } else {
        1
    }
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

#[cfg(test)]
mod tests {
    use super::{
        default_asr_model_spec, find_remote_whisper_assets, find_remote_whisper_weights,
        resolve_model_files, DEFAULT_HF_ASR_MODEL,
    };
    use hf_hub::api::Siblings;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "omnivoice-asr-tests-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn local_whisper_selection_accepts_non_q4_0_gguf_when_needed() {
        let root = unique_temp_dir("gguf-fallback");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();
        fs::write(root.join("tokenizer.json"), "{}").unwrap();
        fs::write(root.join("weights-q4_1.gguf"), b"gguf").unwrap();

        let (_, _, weights, quantized) = resolve_model_files(root.to_str().unwrap()).unwrap();

        assert!(quantized);
        assert_eq!(weights.file_name().unwrap(), "weights-q4_1.gguf");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn local_whisper_selection_prefers_q4_0_gguf_when_present() {
        let root = unique_temp_dir("gguf-prefer-q4_0");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();
        fs::write(root.join("tokenizer.json"), "{}").unwrap();
        fs::write(root.join("weights-q8_0.gguf"), b"gguf").unwrap();
        fs::write(root.join("weights-q4_0.gguf"), b"gguf").unwrap();

        let (_, _, weights, quantized) = resolve_model_files(root.to_str().unwrap()).unwrap();

        assert!(quantized);
        assert_eq!(weights.file_name().unwrap(), "weights-q4_0.gguf");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn local_whisper_selection_falls_back_to_safetensors_without_gguf() {
        let root = unique_temp_dir("safetensors-fallback");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();
        fs::write(root.join("tokenizer.json"), "{}").unwrap();
        fs::write(root.join("model.safetensors"), b"stub").unwrap();

        let (_, _, weights, quantized) = resolve_model_files(root.to_str().unwrap()).unwrap();

        assert!(!quantized);
        assert_eq!(weights.file_name().unwrap(), "model.safetensors");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn remote_whisper_selection_prefers_q4_0_then_other_gguf() {
        let siblings = vec![
            Siblings {
                rfilename: "tokenizer.json".to_string(),
            },
            Siblings {
                rfilename: "weights-q8_0.gguf".to_string(),
            },
            Siblings {
                rfilename: "weights-q4_0.gguf".to_string(),
            },
        ];
        assert_eq!(
            find_remote_whisper_weights(&siblings).unwrap(),
            "weights-q4_0.gguf"
        );

        let fallback_siblings = vec![Siblings {
            rfilename: "weights-q8_0.gguf".to_string(),
        }];
        assert_eq!(
            find_remote_whisper_weights(&fallback_siblings).unwrap(),
            "weights-q8_0.gguf"
        );
    }

    #[test]
    fn default_asr_model_spec_prefers_runtime_local_whisper_bundle() {
        let root = unique_temp_dir("default-local-whisper");
        let whisper_root = root.join("whisper");
        fs::create_dir_all(&whisper_root).unwrap();

        assert_eq!(
            default_asr_model_spec(Some(root.as_path())),
            whisper_root.display().to_string()
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn default_asr_model_spec_falls_back_to_default_repo_without_local_whisper_bundle() {
        let root = unique_temp_dir("default-remote-whisper");
        fs::create_dir_all(&root).unwrap();

        assert_eq!(
            default_asr_model_spec(Some(root.as_path())),
            DEFAULT_HF_ASR_MODEL
        );
        assert_eq!(default_asr_model_spec(None), DEFAULT_HF_ASR_MODEL);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn default_remote_whisper_repo_requires_exact_candle_files() {
        let siblings = vec![
            Siblings {
                rfilename: "config.json".to_string(),
            },
            Siblings {
                rfilename: "tokenizer.json".to_string(),
            },
            Siblings {
                rfilename: "whisper-large-v3-turbo-q4_0.gguf".to_string(),
            },
            Siblings {
                rfilename: "whisper-large-v3-turbo-q8_0.gguf".to_string(),
            },
            Siblings {
                rfilename: "whisper.cpp/whisper-large-v3-turbo-q4_0.gguf".to_string(),
            },
        ];

        let assets = find_remote_whisper_assets(DEFAULT_HF_ASR_MODEL, &siblings).unwrap();

        assert_eq!(assets.config, "config.json");
        assert_eq!(assets.tokenizer, "tokenizer.json");
        assert_eq!(assets.weights, "whisper-large-v3-turbo-q4_0.gguf");
    }

    #[test]
    fn default_remote_whisper_repo_rejects_non_q4_0_fallbacks() {
        let siblings = vec![
            Siblings {
                rfilename: "config.json".to_string(),
            },
            Siblings {
                rfilename: "tokenizer.json".to_string(),
            },
            Siblings {
                rfilename: "whisper-large-v3-turbo-q4_1.gguf".to_string(),
            },
            Siblings {
                rfilename: "whisper-large-v3-turbo-q8_0.gguf".to_string(),
            },
        ];

        let error = find_remote_whisper_assets(DEFAULT_HF_ASR_MODEL, &siblings).unwrap_err();

        assert!(error.to_string().contains("whisper-large-v3-turbo-q4_0.gguf"));
    }
}
