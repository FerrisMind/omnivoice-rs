use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::{
    artifacts::{GeneratorArtifacts, RuntimeArtifacts},
    contracts::{
        BatchedInputs, F32Tensor3, F32Tensor4, I64Tensor2, I64Tensor3, PreparedInferenceBatch,
    },
    error::{OmniVoiceError, Result},
    runtime::RuntimeOptions,
    stage0_loop::{build_timesteps, build_unmask_schedules},
    stage0_qwen3::{Stage0ForwardPass, Stage0Qwen3Backbone, Stage0Qwen3Config},
};

#[derive(Debug, Clone, Deserialize)]
pub struct LocalQwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    pub rope_parameters: RopeParameters,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stage0Config {
    pub audio_vocab_size: usize,
    pub audio_mask_id: i64,
    pub num_audio_codebook: usize,
    pub llm_config: LocalQwen3Config,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stage0DeterministicConfig {
    pub num_step: usize,
    pub guidance_scale: f32,
    pub t_shift: f32,
    pub layer_penalty_factor: f32,
    pub position_temperature: f32,
    pub class_temperature: f32,
    pub capture_steps: Vec<usize>,
    pub capture_layers: Vec<usize>,
    pub capture_final_hidden: bool,
}

impl Default for Stage0DeterministicConfig {
    fn default() -> Self {
        Self {
            num_step: 32,
            guidance_scale: 2.0,
            t_shift: 0.1,
            layer_penalty_factor: 5.0,
            position_temperature: 0.0,
            class_temperature: 0.0,
            capture_steps: vec![0, 15, 31],
            capture_layers: vec![0, 13, 27],
            capture_final_hidden: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stage0StepDebugCapture {
    pub step: usize,
    pub c_logits: F32Tensor4,
    pub u_logits: F32Tensor4,
    pub pred_tokens: I64Tensor3,
    pub confidence_scores: F32Tensor3,
    pub batch_input_ids_before_update: I64Tensor3,
    pub tokens_after_step: I64Tensor3,
    pub batch_input_ids_before_step: I64Tensor3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stage0DebugCapture {
    pub inputs_embeds: F32Tensor3,
    pub hidden_layers: BTreeMap<usize, F32Tensor3>,
    pub final_hidden: F32Tensor3,
    pub steps: Vec<Stage0StepDebugCapture>,
    pub final_tokens: I64Tensor2,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stage0ParityMetric {
    pub exact_match: bool,
    pub max_abs: f32,
    pub mae: f32,
    pub rmse: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stage0ParityMetrics {
    pub metrics: BTreeMap<String, Stage0ParityMetric>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stage0DebugRun {
    pub tokens: I64Tensor2,
    pub debug_capture: Stage0DebugCapture,
    pub parity_metrics: Stage0ParityMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stage0GenerationOutput {
    pub tokens: Vec<I64Tensor2>,
    pub debug_capture: Option<Stage0DebugCapture>,
}

#[derive(Debug)]
struct Stage0DeviceGenerationOutput {
    tokens: Vec<Tensor>,
    debug_capture: Option<Stage0DebugCapture>,
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

impl LocalQwen3Config {
    pub fn hidden_act(&self) -> Result<candle_nn::Activation> {
        match self.hidden_act.as_str() {
            "silu" => Ok(candle_nn::Activation::Silu),
            other => Err(OmniVoiceError::Unsupported(format!(
                "unsupported Qwen3 hidden activation {other}"
            ))),
        }
    }

    fn to_stage0_qwen3_config(&self) -> Result<Stage0Qwen3Config> {
        Ok(Stage0Qwen3Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            attention_bias: self.attention_bias,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_parameters.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            hidden_act: self.hidden_act()?,
        })
    }
}

impl Stage0Config {
    pub fn from_model_root(model_root: impl AsRef<Path>) -> Result<Self> {
        let runtime = RuntimeArtifacts::from_model_root(model_root)?;
        Self::from_artifacts(runtime.generator())
    }

    pub fn from_artifacts(generator: &GeneratorArtifacts) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(
            generator.config_path(),
        )?)?)
    }
}

#[derive(Debug, Clone)]
pub struct Stage0WeightLayout {
    weights_path: PathBuf,
    accepted_prefixes: BTreeSet<String>,
    ignored_keys: BTreeSet<String>,
}

impl Stage0WeightLayout {
    pub fn from_model_root(model_root: impl AsRef<Path>) -> Result<Self> {
        let runtime = RuntimeArtifacts::from_model_root(model_root)?;
        Self::from_artifacts(runtime.generator())
    }

    pub fn from_artifacts(generator: &GeneratorArtifacts) -> Result<Self> {
        Ok(Self {
            weights_path: generator.weights_path().to_path_buf(),
            accepted_prefixes: generator.observed_prefixes().clone(),
            ignored_keys: generator.ignored_keys().clone(),
        })
    }

    pub fn from_safetensors_file(
        path: impl AsRef<Path>,
        required_prefixes: &BTreeSet<String>,
        ignored_keys: &BTreeSet<String>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut accepted_prefixes = BTreeSet::new();
        let mut found_ignored_keys = BTreeSet::new();
        let bytes = fs::read(&path)?;
        let leaked = Box::leak(bytes.into_boxed_slice());
        let tensors = SafeTensors::deserialize(leaked)?;
        for key in tensors.names() {
            if ignored_keys.contains(key) {
                found_ignored_keys.insert(key.to_string());
                continue;
            }
            if let Some(prefix) = key.split('.').next() {
                accepted_prefixes.insert(prefix.to_string());
            }
        }
        if accepted_prefixes != *required_prefixes {
            return Err(OmniVoiceError::InvalidData(format!(
                "stage0 weight prefixes {:?} do not match required {:?}",
                accepted_prefixes, required_prefixes
            )));
        }
        Ok(Self {
            weights_path: path,
            accepted_prefixes,
            ignored_keys: found_ignored_keys,
        })
    }

    pub fn accepted_prefixes(&self) -> &BTreeSet<String> {
        &self.accepted_prefixes
    }

    pub fn ignored_keys(&self) -> &BTreeSet<String> {
        &self.ignored_keys
    }

    pub fn weights_path(&self) -> &Path {
        &self.weights_path
    }
}

#[derive(Debug)]
pub struct Stage0Model {
    backbone: Stage0Qwen3Backbone,
    audio_embeddings: Embedding,
    audio_heads: Linear,
    codebook_layer_offsets: Tensor,
    num_audio_codebook: usize,
    audio_vocab_size: usize,
}

#[derive(Debug)]
pub struct Stage0RuntimePlan {
    options: RuntimeOptions,
    config: Stage0Config,
    weight_layout: Stage0WeightLayout,
    device: Device,
    runtime_dtype: DType,
    model: OnceLock<std::result::Result<Stage0Model, String>>,
}

impl Stage0RuntimePlan {
    pub fn from_options(options: RuntimeOptions) -> Result<Self> {
        let runtime = options.load_runtime_artifacts()?;
        Self::from_runtime_artifacts(options, &runtime)
    }

    pub fn from_runtime_artifacts(
        options: RuntimeOptions,
        runtime: &RuntimeArtifacts,
    ) -> Result<Self> {
        let device = options.resolve_device()?;
        let runtime_dtype = options.resolve_dtype_for_runtime_device(&device);
        Ok(Self {
            config: Stage0Config::from_artifacts(runtime.generator())?,
            weight_layout: Stage0WeightLayout::from_artifacts(runtime.generator())?,
            device,
            runtime_dtype,
            options,
            model: OnceLock::new(),
        })
    }

    pub fn prepare_batch(
        &self,
        batched: &BatchedInputs,
        cond_lens: &[usize],
        target_lens: &[usize],
    ) -> Result<PreparedInferenceBatch> {
        if cond_lens.len() != target_lens.len() {
            return Err(OmniVoiceError::InvalidRequest(format!(
                "cond_lens length {} does not match target_lens length {}",
                cond_lens.len(),
                target_lens.len()
            )));
        }

        Ok(PreparedInferenceBatch {
            input_ids: batched.batch_input_ids.to_candle(&self.device)?,
            audio_mask: batched.batch_audio_mask.to_candle(&self.device)?,
            attention_mask: batched.batch_attention_mask.to_candle(&self.device)?,
            tokens_init: batched.tokens_init.to_candle(&self.device)?,
            target_lens: target_lens.to_vec(),
            cond_lens: cond_lens.to_vec(),
            runtime_dtype: self.runtime_dtype,
        })
    }

    pub fn generate_deterministic(
        &self,
        prepared: &PreparedInferenceBatch,
        config: &Stage0DeterministicConfig,
        capture_steps: &[usize],
    ) -> Result<Stage0GenerationOutput> {
        self.run_loop(prepared, config, capture_steps)
    }

    pub fn generate_deterministic_device(
        &self,
        prepared: &PreparedInferenceBatch,
        config: &Stage0DeterministicConfig,
    ) -> Result<Vec<Tensor>> {
        Ok(self.run_loop_device(prepared, config, &[])?.tokens)
    }

    pub fn debug_case(
        &self,
        prepared: &PreparedInferenceBatch,
        config: &Stage0DeterministicConfig,
    ) -> Result<Stage0DebugCapture> {
        self.generate_deterministic(prepared, config, &config.capture_steps)?
            .debug_capture
            .ok_or_else(|| {
                OmniVoiceError::InvalidData(
                    "stage0 deterministic debug run did not capture debug tensors".to_string(),
                )
            })
    }

    pub fn config(&self) -> &Stage0Config {
        &self.config
    }

    pub fn weight_layout(&self) -> &Stage0WeightLayout {
        &self.weight_layout
    }

    pub fn options(&self) -> &RuntimeOptions {
        &self.options
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn runtime_dtype(&self) -> DType {
        self.runtime_dtype
    }

    pub fn is_loaded(&self) -> bool {
        self.model.get().is_some()
    }

    fn model(&self) -> Result<&Stage0Model> {
        let result = self.model.get_or_init(|| {
            Stage0Model::load(
                &self.config,
                self.weight_layout.weights_path(),
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

    fn run_loop(
        &self,
        prepared: &PreparedInferenceBatch,
        config: &Stage0DeterministicConfig,
        capture_steps: &[usize],
    ) -> Result<Stage0GenerationOutput> {
        let output = self.run_loop_device(prepared, config, capture_steps)?;
        Ok(Stage0GenerationOutput {
            tokens: output
                .tokens
                .iter()
                .map(tensor_to_i64_tensor2)
                .collect::<Result<Vec<_>>>()?,
            debug_capture: output.debug_capture,
        })
    }

    fn run_loop_device(
        &self,
        prepared: &PreparedInferenceBatch,
        config: &Stage0DeterministicConfig,
        capture_steps: &[usize],
    ) -> Result<Stage0DeviceGenerationOutput> {
        if let Some(seed) = self.options.seed() {
            self.device.set_seed(seed)?;
        }
        let batch_size = prepared.target_lens.len();
        let debug_enabled = !capture_steps.is_empty();
        if batch_size != 1 && debug_enabled {
            return Err(OmniVoiceError::Unsupported(
                "stage0 debug capture currently supports one prompt per run".to_string(),
            ));
        }

        let model = self.model()?;
        let num_codebooks = self.config.num_audio_codebook;
        let capture_layers: BTreeSet<usize> = config.capture_layers.iter().copied().collect();
        let capture_steps: BTreeSet<usize> = capture_steps.iter().copied().collect();
        let attention_mask =
            model.prepare_attention_mask(&prepared.attention_mask, self.runtime_dtype)?;
        let timesteps = build_timesteps(0.0, 1.0, config.num_step + 1, config.t_shift)?;
        let schedules = build_unmask_schedules(
            &prepared.target_lens,
            num_codebooks,
            &timesteps,
            config.num_step,
        )?;
        let layer_penalties = Tensor::from_vec(
            (0..num_codebooks)
                .map(|layer| layer as f32 * config.layer_penalty_factor)
                .collect::<Vec<_>>(),
            (1, num_codebooks, 1),
            &self.device,
        )?;
        let mut batch_input_ids = prepared.input_ids.clone();
        let mut tokens = prepared.tokens_init.clone();

        let initial_forward = if debug_enabled {
            Some(model.forward(
                &batch_input_ids,
                &prepared.audio_mask,
                &attention_mask,
                &capture_layers,
            )?)
        } else {
            None
        };

        let mut step_captures = Vec::new();
        for step in 0..config.num_step {
            let forward = model.forward(
                &batch_input_ids,
                &prepared.audio_mask,
                &attention_mask,
                &BTreeSet::new(),
            )?;
            let batch_logits = forward.logits.to_dtype(DType::F32)?;
            for (batch_index, batch_schedule) in schedules.iter().enumerate().take(batch_size) {
                let target_len = prepared.target_lens[batch_index];
                let cond_len = prepared.cond_lens[batch_index];
                let c_logits = batch_logits.i((
                    batch_index..batch_index + 1,
                    ..,
                    (cond_len - target_len)..cond_len,
                    ..,
                ))?;
                let u_logits = batch_logits.i((
                    batch_size + batch_index..batch_size + batch_index + 1,
                    ..,
                    0..target_len,
                    ..,
                ))?;
                let batch_input_ids_before_update =
                    if debug_enabled && capture_steps.contains(&step) {
                        Some(batch_input_ids.clone())
                    } else {
                        None
                    };
                let (pred_tokens_tensor, confidence_scores_tensor) =
                    predict_tokens_with_scoring_from_tensors(
                        &c_logits,
                        &u_logits,
                        config.guidance_scale,
                        config.class_temperature,
                        self.config.audio_mask_id as usize,
                    )?;
                let current_tokens_view =
                    tokens.i((batch_index..batch_index + 1, .., 0..target_len))?;
                let updated_tokens = apply_step_updates_device(
                    &current_tokens_view,
                    &pred_tokens_tensor,
                    &confidence_scores_tensor,
                    self.config.audio_mask_id,
                    batch_schedule[step],
                    &layer_penalties,
                    config.position_temperature,
                )?;
                tokens = tokens.slice_assign(
                    &[
                        batch_index..batch_index + 1,
                        0..num_codebooks,
                        0..target_len,
                    ],
                    &updated_tokens,
                )?;
                batch_input_ids = batch_input_ids.slice_assign(
                    &[
                        batch_index..batch_index + 1,
                        0..num_codebooks,
                        (cond_len - target_len)..cond_len,
                    ],
                    &updated_tokens,
                )?;
                batch_input_ids = batch_input_ids.slice_assign(
                    &[
                        batch_size + batch_index..batch_size + batch_index + 1,
                        0..num_codebooks,
                        0..target_len,
                    ],
                    &updated_tokens,
                )?;

                if debug_enabled && capture_steps.contains(&step) {
                    step_captures.push(Stage0StepDebugCapture {
                        step,
                        c_logits: tensor_to_f32_tensor4(&c_logits)?,
                        u_logits: tensor_to_f32_tensor4(&u_logits)?,
                        pred_tokens: tensor_to_i64_tensor3(&pred_tokens_tensor)?,
                        confidence_scores: tensor_to_f32_tensor3(&confidence_scores_tensor)?,
                        batch_input_ids_before_update: tensor_to_i64_tensor3(
                            batch_input_ids_before_update
                                .as_ref()
                                .expect("capture gate checked"),
                        )?,
                        tokens_after_step: tensor_to_i64_tensor3(&updated_tokens)?,
                        batch_input_ids_before_step: tensor_to_i64_tensor3(&batch_input_ids)?,
                    });
                }
            }
        }

        let mut final_tokens = Vec::with_capacity(batch_size);
        for (batch_index, target_len) in prepared.target_lens.iter().copied().enumerate() {
            final_tokens.push(tokens.i((batch_index, .., 0..target_len))?);
        }

        let debug_capture = if !debug_enabled {
            None
        } else {
            let initial_forward = initial_forward.expect("debug capture checked");
            let mut hidden_layers = BTreeMap::new();
            for (index, tensor) in initial_forward.backbone.captured_hidden_layers.into_iter() {
                hidden_layers.insert(index, tensor_to_f32_tensor3(&tensor)?);
            }
            Some(Stage0DebugCapture {
                inputs_embeds: tensor_to_f32_tensor3(&initial_forward.inputs_embeds)?,
                hidden_layers,
                final_hidden: tensor_to_f32_tensor3(&initial_forward.backbone.final_hidden)?,
                steps: step_captures,
                final_tokens: tensor_to_i64_tensor2(
                    final_tokens
                        .first()
                        .expect("debug capture requires one prompt"),
                )?,
            })
        };

        Ok(Stage0DeviceGenerationOutput {
            tokens: final_tokens,
            debug_capture,
        })
    }
}

#[derive(Debug)]
pub struct Stage0ForwardOutputs {
    pub inputs_embeds: Tensor,
    pub backbone: Stage0ForwardPass,
    pub logits: Tensor,
}

impl Stage0Model {
    fn load(
        config: &Stage0Config,
        weights_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = mmap_var_builder(weights_path, dtype, device)?;
        let llm_vb = vb.clone().rename_f(|name| {
            if let Some(stripped) = name.strip_prefix("model.") {
                format!("llm.{stripped}")
            } else {
                format!("llm.{name}")
            }
        });
        let backbone =
            Stage0Qwen3Backbone::load(&config.llm_config.to_stage0_qwen3_config()?, llm_vb)?;
        let audio_embeddings_weight = vb.get(
            (
                config.num_audio_codebook * config.audio_vocab_size,
                config.llm_config.hidden_size,
            ),
            "audio_embeddings.weight",
        )?;
        let audio_embeddings =
            Embedding::new(audio_embeddings_weight, config.llm_config.hidden_size);
        let audio_heads_weight = vb.get(
            (
                config.num_audio_codebook * config.audio_vocab_size,
                config.llm_config.hidden_size,
            ),
            "audio_heads.weight",
        )?;
        let audio_heads = Linear::new(audio_heads_weight, None);
        let codebook_layer_offsets = Tensor::from_vec(
            AudioEmbeddingMixer::new(config.num_audio_codebook, config.audio_vocab_size)
                .layer_offsets(),
            (1, config.num_audio_codebook, 1),
            device,
        )?;
        Ok(Self {
            backbone,
            audio_embeddings,
            audio_heads,
            codebook_layer_offsets,
            num_audio_codebook: config.num_audio_codebook,
            audio_vocab_size: config.audio_vocab_size,
        })
    }

    fn prepare_attention_mask(&self, mask: &Tensor, dtype: DType) -> Result<Tensor> {
        let dims = mask.dims();
        if dims.len() != 4 {
            return Err(OmniVoiceError::InvalidTensorShape {
                name: "stage0_attention_mask".to_string(),
                expected: "(B, 1, Q, K)".to_string(),
                actual: format!("{dims:?}"),
            });
        }
        let on_true = Tensor::zeros(mask.shape(), dtype, mask.device())?;
        let on_false =
            Tensor::new(f32::NEG_INFINITY, mask.device())?.broadcast_as(mask.shape().dims())?;
        mask.where_cond(&on_true, &on_false.to_dtype(dtype)?)
            .map_err(Into::into)
    }

    fn prepare_embed_inputs(&self, input_ids: &Tensor, audio_mask: &Tensor) -> Result<Tensor> {
        let text_ids = input_ids.i((.., 0, ..))?;
        let text_embeds = self.backbone.embed_text_tokens(&text_ids)?;
        let shifted_ids = (input_ids
            .broadcast_mul(&audio_mask.unsqueeze(1)?.to_dtype(DType::I64)?)?
            + self
                .codebook_layer_offsets
                .broadcast_as(input_ids.shape().dims())?)?;
        let audio_embeds = self.audio_embeddings.forward(&shifted_ids)?.sum(1)?;
        let selection_mask = audio_mask
            .unsqueeze(candle_core::D::Minus1)?
            .broadcast_as(audio_embeds.shape().dims())?;
        selection_mask
            .where_cond(&audio_embeds, &text_embeds)
            .map_err(Into::into)
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        audio_mask: &Tensor,
        attention_mask: &Tensor,
        capture_layers: &BTreeSet<usize>,
    ) -> Result<Stage0ForwardOutputs> {
        let inputs_embeds = self.prepare_embed_inputs(input_ids, audio_mask)?;
        let backbone =
            self.backbone
                .forward_embeds(&inputs_embeds, Some(attention_mask), capture_layers)?;
        let final_hidden = backbone.final_hidden.clone();
        let (batch_size, seq_len, _) = final_hidden.dims3()?;
        let logits = self
            .audio_heads
            .forward(&final_hidden)?
            .reshape((
                batch_size,
                seq_len,
                self.num_audio_codebook,
                self.audio_vocab_size,
            ))?
            .permute((0, 2, 1, 3))?;
        Ok(Stage0ForwardOutputs {
            inputs_embeds,
            backbone,
            logits,
        })
    }
}

#[derive(Debug, Clone)]
pub struct AudioEmbeddingMixer {
    num_audio_codebook: usize,
    audio_vocab_size: usize,
}

impl AudioEmbeddingMixer {
    pub fn new(num_audio_codebook: usize, audio_vocab_size: usize) -> Self {
        Self {
            num_audio_codebook,
            audio_vocab_size,
        }
    }

    pub fn layer_offsets(&self) -> Vec<i64> {
        (0..self.num_audio_codebook)
            .map(|index| (index * self.audio_vocab_size) as i64)
            .collect()
    }

    pub fn shifted_audio_id(&self, layer: usize, token_id: i64) -> Result<i64> {
        let Some(offset) = self.layer_offsets().get(layer).copied() else {
            return Err(OmniVoiceError::InvalidRequest(format!(
                "layer {layer} out of range for {} codebooks",
                self.num_audio_codebook
            )));
        };
        Ok(token_id + offset)
    }
}

fn predict_tokens_with_scoring_from_tensors(
    c_logits: &Tensor,
    u_logits: &Tensor,
    guidance_scale: f32,
    class_temperature: f32,
    audio_mask_id: usize,
) -> Result<(Tensor, Tensor)> {
    let log_probs = if guidance_scale != 0.0 {
        let c_log_probs = candle_nn::ops::log_softmax(c_logits, candle_core::D::Minus1)?;
        let u_log_probs = candle_nn::ops::log_softmax(u_logits, candle_core::D::Minus1)?;
        let guided = (&c_log_probs + ((&c_log_probs - &u_log_probs)? * guidance_scale as f64)?)?;
        candle_nn::ops::log_softmax(&guided, candle_core::D::Minus1)?
    } else {
        candle_nn::ops::log_softmax(c_logits, candle_core::D::Minus1)?
    };
    let log_probs = mask_audio_token(&log_probs, audio_mask_id)?;
    let pred_tokens = if class_temperature > 0.0 {
        let filtered = filter_top_k(&log_probs, 0.1)?;
        gumbel_argmax(&filtered, class_temperature)?
    } else {
        log_probs.argmax(candle_core::D::Minus1)?
    }
    .to_dtype(DType::I64)?;
    let confidence_scores = log_probs.max(candle_core::D::Minus1)?;
    Ok((pred_tokens, confidence_scores))
}

fn mask_audio_token(log_probs: &Tensor, audio_mask_id: usize) -> Result<Tensor> {
    let vocab_size = log_probs.dim(candle_core::D::Minus1)?;
    if audio_mask_id >= vocab_size {
        return Err(OmniVoiceError::InvalidRequest(format!(
            "audio mask token {audio_mask_id} is outside vocab size {vocab_size}"
        )));
    }
    let prefix = log_probs.narrow(candle_core::D::Minus1, 0, audio_mask_id)?;
    let mask_fill = Tensor::full(
        f32::NEG_INFINITY,
        prefix
            .shape()
            .dims()
            .iter()
            .copied()
            .take(prefix.rank().saturating_sub(1))
            .chain(std::iter::once(1))
            .collect::<Vec<_>>(),
        log_probs.device(),
    )?;
    if audio_mask_id + 1 == vocab_size {
        Tensor::cat(&[&prefix, &mask_fill], candle_core::D::Minus1).map_err(Into::into)
    } else {
        let suffix = log_probs.narrow(
            candle_core::D::Minus1,
            audio_mask_id + 1,
            vocab_size - audio_mask_id - 1,
        )?;
        Tensor::cat(&[&prefix, &mask_fill, &suffix], candle_core::D::Minus1).map_err(Into::into)
    }
}

fn filter_top_k(logits: &Tensor, ratio: f32) -> Result<Tensor> {
    let vocab_size = logits.dim(candle_core::D::Minus1)?;
    let top_k = ((ratio * vocab_size as f32).ceil() as usize).clamp(1, vocab_size);
    let sorted_indices = logits.arg_sort_last_dim(false)?;
    let top_indices = sorted_indices.narrow(candle_core::D::Minus1, 0, top_k)?;
    let top_values = logits.gather(&top_indices, candle_core::D::Minus1)?;
    let masked = Tensor::full(f32::NEG_INFINITY, logits.shape().dims(), logits.device())?;
    masked
        .scatter(&top_indices, &top_values, candle_core::D::Minus1)
        .map_err(Into::into)
}

fn gumbel_argmax(logits: &Tensor, temperature: f32) -> Result<Tensor> {
    if temperature <= 0.0 {
        return logits.argmax(candle_core::D::Minus1).map_err(Into::into);
    }
    let logits = logits.to_dtype(DType::F32)?;
    let uniform = logits.rand_like(0f64, 1f64)?;
    let gumbel_noise = (&uniform + 1e-10)?
        .log()?
        .neg()?
        .broadcast_add(&Tensor::new(1e-10f32, uniform.device())?)?
        .log()?
        .neg()?;
    ((&logits / temperature as f64)? + gumbel_noise)?
        .argmax(candle_core::D::Minus1)
        .map_err(Into::into)
}

fn apply_position_temperature(logits: &Tensor, temperature: f32) -> Result<Tensor> {
    if temperature <= 0.0 {
        return Ok(logits.clone());
    }
    let logits = logits.to_dtype(DType::F32)?;
    let uniform = logits.rand_like(0f64, 1f64)?;
    let gumbel_noise = (&uniform + 1e-10)?
        .log()?
        .neg()?
        .broadcast_add(&Tensor::new(1e-10f32, uniform.device())?)?
        .log()?
        .neg()?;
    ((&logits / temperature as f64)? + gumbel_noise).map_err(Into::into)
}

fn apply_step_updates_device(
    current_tokens: &Tensor,
    predicted_tokens: &Tensor,
    confidence_scores: &Tensor,
    mask_id: i64,
    update_count: usize,
    layer_penalties: &Tensor,
    position_temperature: f32,
) -> Result<Tensor> {
    if update_count == 0 {
        return Ok(current_tokens.clone());
    }
    let selection_scores = confidence_scores
        .broadcast_sub(&layer_penalties.broadcast_as(confidence_scores.shape().dims())?)?;
    let selection_scores = apply_position_temperature(&selection_scores, position_temperature)?;
    let available_mask = current_tokens.eq(mask_id)?;
    let neg_inf = Tensor::full(
        f32::NEG_INFINITY,
        selection_scores.shape().dims(),
        current_tokens.device(),
    )?;
    let masked_scores = available_mask.where_cond(&selection_scores, &neg_inf)?;
    let flat_scores = masked_scores.flatten_all()?;
    let flat_len = flat_scores.elem_count();
    let top_k = update_count.min(flat_len);
    let sorted_indices = flat_scores
        .reshape((1, flat_len))?
        .arg_sort_last_dim(false)?;
    let top_indices = sorted_indices.i((0, 0..top_k))?;
    let flat_predicted = predicted_tokens.flatten_all()?;
    let update_values = flat_predicted.gather(&top_indices, 0)?;
    let flat_current = current_tokens.flatten_all()?;
    let updated = if current_tokens.device().is_metal() && current_tokens.dtype() == DType::I64 {
        let flat_current = flat_current.to_dtype(DType::U32)?;
        let update_values = update_values.to_dtype(DType::U32)?;
        flat_current
            .scatter(&top_indices, &update_values, 0)?
            .to_dtype(DType::I64)?
    } else {
        flat_current.scatter(&top_indices, &update_values, 0)?
    };
    updated
        .reshape(current_tokens.shape().dims())
        .map_err(Into::into)
}

fn tensor_parity_metric(actual: &[f32], expected: &[f32]) -> Result<Stage0ParityMetric> {
    if actual.len() != expected.len() {
        return Err(OmniVoiceError::InvalidData(format!(
            "tensor parity length mismatch: actual len {} != expected len {}",
            actual.len(),
            expected.len()
        )));
    }
    let mut max_abs = 0.0_f32;
    let mut abs_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut exact_match = true;
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        let diff = *lhs - *rhs;
        if diff != 0.0 {
            exact_match = false;
        }
        max_abs = max_abs.max(diff.abs());
        abs_sum += f64::from(diff.abs());
        squared_sum += f64::from(diff * diff);
    }
    let count = actual.len().max(1) as f64;
    Ok(Stage0ParityMetric {
        exact_match,
        max_abs,
        mae: (abs_sum / count) as f32,
        rmse: (squared_sum / count).sqrt() as f32,
    })
}

fn exact_i64_parity_metric(actual: &[i64], expected: &[i64]) -> Stage0ParityMetric {
    Stage0ParityMetric {
        exact_match: actual == expected,
        max_abs: if actual == expected {
            0.0
        } else {
            f32::INFINITY
        },
        mae: if actual == expected {
            0.0
        } else {
            f32::INFINITY
        },
        rmse: if actual == expected {
            0.0
        } else {
            f32::INFINITY
        },
    }
}

impl Stage0ParityMetrics {
    pub fn from_debug_capture(
        actual: &Stage0DebugCapture,
        reference_forward: &crate::artifacts::ForwardStepZero,
        reference_steps: &[(usize, crate::artifacts::StepCapture)],
        reference_final_tokens: &I64Tensor2,
    ) -> Result<Self> {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "inputs_embeds".to_string(),
            tensor_parity_metric(
                &actual.inputs_embeds.data,
                &reference_forward.inputs_embeds.data,
            )?,
        );
        for (layer, reference_hidden) in &reference_forward.hidden_layers {
            let actual_hidden = actual.hidden_layers.get(layer).ok_or_else(|| {
                OmniVoiceError::InvalidData(format!(
                    "missing actual hidden layer {:02} capture",
                    layer
                ))
            })?;
            metrics.insert(
                format!("hidden_layer_{layer:02}"),
                tensor_parity_metric(&actual_hidden.data, &reference_hidden.data)?,
            );
        }
        metrics.insert(
            "final_hidden".to_string(),
            tensor_parity_metric(
                &actual.final_hidden.data,
                &reference_forward.final_hidden.data,
            )?,
        );
        metrics.insert(
            "final_tokens".to_string(),
            exact_i64_parity_metric(&actual.final_tokens.data, &reference_final_tokens.data),
        );
        for (step, reference) in reference_steps {
            let actual_step = actual
                .steps
                .iter()
                .find(|capture| capture.step == *step)
                .ok_or_else(|| {
                    OmniVoiceError::InvalidData(format!(
                        "missing actual stage0 debug capture for step {step}"
                    ))
                })?;
            metrics.insert(
                format!("step_{step:02}_c_logits"),
                tensor_parity_metric(&actual_step.c_logits.data, &reference.c_logits.data)?,
            );
            metrics.insert(
                format!("step_{step:02}_u_logits"),
                tensor_parity_metric(&actual_step.u_logits.data, &reference.u_logits.data)?,
            );
            metrics.insert(
                format!("step_{step:02}_pred_tokens"),
                exact_i64_parity_metric(&actual_step.pred_tokens.data, &reference.pred_tokens.data),
            );
            metrics.insert(
                format!("step_{step:02}_confidence_scores"),
                tensor_parity_metric(
                    &actual_step.confidence_scores.data,
                    &reference.confidence_scores.data,
                )?,
            );
            metrics.insert(
                format!("step_{step:02}_batch_input_ids_before_step"),
                exact_i64_parity_metric(
                    &actual_step.batch_input_ids_before_step.data,
                    &reference.batch_input_ids_before_step.data,
                ),
            );
            metrics.insert(
                format!("step_{step:02}_tokens_after_step"),
                exact_i64_parity_metric(
                    &actual_step.tokens_after_step.data,
                    &reference.tokens_after_step.data,
                ),
            );
        }
        Ok(Self { metrics })
    }

    pub fn insert_exact_i64_metric(
        &mut self,
        name: impl Into<String>,
        actual: &[i64],
        expected: &[i64],
    ) {
        self.metrics
            .insert(name.into(), exact_i64_parity_metric(actual, expected));
    }
}

fn mmap_var_builder(
    weights_path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let paths = [weights_path];
    // SAFETY: Candle exposes mmap loading behind an unsafe API because it wraps OS-backed
    // read-only memory maps. We map immutable safetensors files and only hand the backend to
    // Candle for read-only tensor materialization.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? })
}

fn tensor_to_f32_tensor3(tensor: &Tensor) -> Result<F32Tensor3> {
    let dims = tensor.dims3()?;
    let data = tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor3::new(dims, data)
}

fn tensor_to_f32_tensor4(tensor: &Tensor) -> Result<F32Tensor4> {
    let dims = tensor.dims4()?;
    let data = tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor4::new(dims, data)
}

fn tensor_to_i64_tensor3(tensor: &Tensor) -> Result<I64Tensor3> {
    let dims = tensor.dims3()?;
    let data = tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    I64Tensor3::new(dims, data)
}

pub(crate) fn tensor_to_i64_tensor2(tensor: &Tensor) -> Result<I64Tensor2> {
    let dims = tensor.dims2()?;
    let data = tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    I64Tensor2::new(dims, data)
}
