use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};

use crate::{
    artifacts::{ReferenceArtifactBundle, RuntimeArtifacts},
    asr::WhisperAsr,
    audio_input::ReferenceAudioProcessor,
    audio_tokenizer::AudioTokenizerRuntimePlan,
    contracts::{
        DecodedAudio, GeneratedAudioResult, GeneratedTokens, GenerationRequest, GenerationUsage,
        PreparedInferenceBatch, PreparedPrompt, PreparedPromptSequence, ReferenceAudioInput,
        VoiceClonePrompt, WaveformInput,
    },
    error::Result,
    frontend::{
        add_punctuation, DeviceGenerationTask, DeviceVoiceClonePrompt, Frontend,
        PreparedPromptDevice,
    },
    reference_prompt::{ReferencePromptBuilder, ReferencePromptOptions},
    runtime::RuntimeOptions,
    stage0_loop::pack_cfg_batch,
    stage0_model::{
        tensor_to_i64_tensor2, Stage0DebugRun, Stage0DeterministicConfig, Stage0RuntimePlan,
    },
    stage1_decoder::{PreparedStage1Decode, Stage1DebugRun, Stage1RuntimePlan},
};

#[derive(Clone)]
enum GeneratedTensorTokens {
    Single(Tensor),
    Chunked(Vec<Tensor>),
}

struct MaterializedDeviceRequest {
    request: GenerationRequest,
    device_voice_clone_prompts: Vec<Option<DeviceVoiceClonePrompt>>,
}

#[derive(Debug)]
pub struct Phase3Pipeline {
    runtime_artifacts: RuntimeArtifacts,
    frontend: Frontend,
    asr: Mutex<Option<(String, WhisperAsr)>>,
    audio_tokenizer: AudioTokenizerRuntimePlan,
    stage0: Stage0RuntimePlan,
    stage1: Stage1RuntimePlan,
}

impl Phase3Pipeline {
    pub fn from_options(options: RuntimeOptions) -> Result<Self> {
        let runtime_artifacts = options.load_runtime_artifacts()?;
        let frontend = Frontend::from_runtime_artifacts(&runtime_artifacts)?;
        let shared_device = options.resolve_device()?;
        let audio_tokenizer = AudioTokenizerRuntimePlan::from_runtime_artifacts_with_device(
            options.clone(),
            &runtime_artifacts,
            shared_device.clone(),
        )?;
        let stage0 = Stage0RuntimePlan::from_runtime_artifacts_with_device(
            options.clone(),
            &runtime_artifacts,
            shared_device.clone(),
        )?;
        let stage1 = Stage1RuntimePlan::from_runtime_artifacts_with_device(
            options.clone(),
            &runtime_artifacts,
            crate::stage1_decoder::Stage1DecoderBundle::from_runtime_artifacts(&runtime_artifacts)?,
            shared_device,
        )?;

        Ok(Self {
            runtime_artifacts,
            frontend,
            asr: Mutex::new(None),
            audio_tokenizer,
            stage0,
            stage1,
        })
    }

    pub fn load_asr_model(&self, model_name: Option<&str>) -> Result<()> {
        let requested = model_name
            .map(str::to_string)
            .unwrap_or_else(crate::asr::default_asr_model_spec);
        let mut guard = self.asr.lock().unwrap_or_else(|poison| poison.into_inner());
        let needs_reload = guard
            .as_ref()
            .map(|(loaded, _)| loaded != &requested)
            .unwrap_or(true);
        if needs_reload {
            *guard = Some((
                requested.clone(),
                WhisperAsr::load(&requested, self.stage0.device().clone())?,
            ));
        }
        Ok(())
    }

    pub fn transcribe(
        &self,
        ref_audio: &ReferenceAudioInput,
        model_name: Option<&str>,
    ) -> Result<String> {
        let processor = ReferenceAudioProcessor::new(
            self.runtime_artifacts.contracts().sample_rate,
            self.runtime_artifacts.contracts().hop_length,
        );
        let waveform = processor.load_input(ref_audio)?;
        let samples = crate::audio_input::mono_samples(&waveform.samples, waveform.channels);
        self.load_asr_model(model_name)?;
        let mut guard = self.asr.lock().unwrap_or_else(|poison| poison.into_inner());
        let (_, asr) = guard.as_mut().ok_or_else(|| {
            crate::error::OmniVoiceError::InvalidData("ASR model is not loaded".to_string())
        })?;
        asr.transcribe(&samples, waveform.sample_rate)
    }

    fn transcribe_waveform(
        &self,
        waveform: &WaveformInput,
        model_name: Option<&str>,
    ) -> Result<String> {
        let samples = crate::audio_input::mono_samples(&waveform.samples, waveform.channels);
        self.load_asr_model(model_name)?;
        let mut guard = self.asr.lock().unwrap_or_else(|poison| poison.into_inner());
        let (_, asr) = guard.as_mut().ok_or_else(|| {
            crate::error::OmniVoiceError::InvalidData("ASR model is not loaded".to_string())
        })?;
        asr.transcribe(&samples, waveform.sample_rate)
    }

    pub fn create_voice_clone_prompt_from_audio(
        &self,
        ref_audio: &ReferenceAudioInput,
        ref_text: Option<&str>,
        preprocess_prompt: bool,
        asr_model: Option<&str>,
    ) -> Result<VoiceClonePrompt> {
        let processor = ReferenceAudioProcessor::new(
            self.runtime_artifacts.contracts().sample_rate,
            self.runtime_artifacts.contracts().hop_length,
        );
        let prepared = processor.prepare_prompt_audio(ref_audio, ref_text, preprocess_prompt)?;
        let resolved_ref_text = match prepared.ref_text {
            Some(ref_text) => ref_text,
            None => {
                let resolved_ref_text = self.transcribe_waveform(
                    &WaveformInput::mono(
                        prepared.waveform.clone(),
                        self.runtime_artifacts.contracts().sample_rate,
                    ),
                    asr_model,
                )?;
                if preprocess_prompt {
                    add_punctuation(&resolved_ref_text)
                } else {
                    resolved_ref_text
                }
            }
        };
        let tokens = self.audio_tokenizer.encode_waveform(
            &prepared.waveform,
            self.runtime_artifacts.contracts().sample_rate,
        )?;

        ReferencePromptBuilder::new(ReferencePromptOptions {
            sample_rate: self.runtime_artifacts.contracts().sample_rate,
            hop_length: self.runtime_artifacts.contracts().hop_length,
            expected_codebooks: self.runtime_artifacts.contracts().num_audio_codebooks,
        })
        .prompt_from_tokens(tokens, resolved_ref_text, prepared.ref_rms)
    }

    pub fn prepare_prompt(&self, request: &GenerationRequest) -> Result<PreparedInferenceBatch> {
        let request = self.materialize_request(request)?;
        let task = self.frontend.build_task(&request)?;
        let mut prepared = Vec::with_capacity(task.batch_size());
        let mut cond_lens = Vec::with_capacity(task.batch_size());
        for index in 0..task.batch_size() {
            let prompt = self.frontend.prepare_prompt(&task, index)?;
            cond_lens.push(prompt.total_length);
            prepared.push(prompt);
        }
        let batched = pack_cfg_batch(&prepared, task.target_lens())?;
        self.stage0
            .prepare_batch(&batched, &cond_lens, task.target_lens())
    }

    pub fn generate_tokens(&self, request: &GenerationRequest) -> Result<Vec<GeneratedTokens>> {
        let request = self.materialize_request(request)?;
        let task = self.frontend.build_task(&request)?;
        self.generate_tokens_from_task(&task)
    }

    pub fn generate(&self, request: &GenerationRequest) -> Result<Vec<DecodedAudio>> {
        Ok(self
            .generate_with_usage(request)?
            .into_iter()
            .map(|result| result.audio)
            .collect())
    }

    pub fn generate_with_usage(
        &self,
        request: &GenerationRequest,
    ) -> Result<Vec<GeneratedAudioResult>> {
        let materialized = self.materialize_device_request(request)?;
        let task = self.frontend.build_task_with_device_prompts(
            &materialized.request,
            &materialized.device_voice_clone_prompts,
        )?;
        let usage = self.estimate_generation_usage(&materialized.request, &task)?;
        let audio = self.generate_audio_from_device_task(&task)?;

        Ok(audio
            .into_iter()
            .zip(usage)
            .map(|(audio, usage)| GeneratedAudioResult { audio, usage })
            .collect())
    }

    pub fn generate_stage0_from_reference_case(
        &self,
        reference_root: impl AsRef<std::path::Path>,
        case_id: &str,
    ) -> Result<GeneratedTokens> {
        let prepared = self.load_prepared_prompts_from_reference_case(reference_root, case_id)?;
        self.generate_stage0_from_prepared_prompts(&prepared)
    }

    pub fn debug_stage0_from_reference_case(
        &self,
        reference_root: impl AsRef<std::path::Path>,
        case_id: &str,
    ) -> Result<Stage0DebugRun> {
        let bundle = ReferenceArtifactBundle::from_root(reference_root)?;
        let case = bundle.case_by_id(case_id)?;
        let prepared = match case.load_prepared_prompts()? {
            PreparedPromptSequence::Single(prompt) => prompt,
            PreparedPromptSequence::Chunked(chunked) => {
                chunked.prompts.last().cloned().ok_or_else(|| {
                    crate::error::OmniVoiceError::InvalidData(
                        "chunked debug case does not contain any prompts".to_string(),
                    )
                })?
            }
        };
        let reference_config = case.load_stage0_debug_config()?;
        let batch = self.prepare_stage0_prompt(&prepared)?;
        let debug_capture = self.stage0.debug_case(
            &batch,
            &Stage0DeterministicConfig {
                num_step: reference_config.num_step,
                guidance_scale: reference_config.guidance_scale,
                t_shift: reference_config.t_shift,
                layer_penalty_factor: reference_config.layer_penalty_factor,
                position_temperature: 0.0,
                class_temperature: 0.0,
                capture_steps: reference_config.capture_steps.clone(),
                capture_layers: reference_config.capture_hidden_layers.clone(),
                capture_final_hidden: reference_config.capture_final_hidden,
            },
        )?;
        let reference_forward = case.load_forward_step_zero()?;
        let reference_steps = reference_config
            .capture_steps
            .iter()
            .copied()
            .map(|step| case.load_step_capture(step).map(|capture| (step, capture)))
            .collect::<Result<Vec<_>>>()?;
        let reference_final_tokens = match case.load_generated_tokens()? {
            GeneratedTokens::Single(tokens) => tokens,
            GeneratedTokens::Chunked(chunks) => chunks.last().cloned().ok_or_else(|| {
                crate::error::OmniVoiceError::InvalidData(
                    "chunked debug case does not contain any token chunks".to_string(),
                )
            })?,
        };
        Ok(Stage0DebugRun {
            tokens: debug_capture.final_tokens.clone(),
            parity_metrics: crate::stage0_model::Stage0ParityMetrics::from_debug_capture(
                &debug_capture,
                &reference_forward,
                &reference_steps,
                &reference_final_tokens,
            )?,
            debug_capture,
        })
    }

    pub fn prepare_prompt_from_reference_case(
        &self,
        reference_root: impl AsRef<std::path::Path>,
        case_id: &str,
    ) -> Result<PreparedInferenceBatch> {
        let bundle = ReferenceArtifactBundle::from_root(reference_root)?;
        let case = bundle.case_by_id(case_id)?;
        let request = case.load_case_definition()?.request;
        self.prepare_prompt(&request)
    }

    pub fn load_prepared_prompts_from_reference_case(
        &self,
        reference_root: impl AsRef<std::path::Path>,
        case_id: &str,
    ) -> Result<PreparedPromptSequence> {
        let bundle = ReferenceArtifactBundle::from_root(reference_root)?;
        let case = bundle.case_by_id(case_id)?;
        case.load_prepared_prompts()
    }

    pub fn prepare_stage1_from_reference_case(
        &self,
        reference_root: impl AsRef<std::path::Path>,
        case_id: &str,
    ) -> Result<PreparedStage1Decode> {
        let bundle = ReferenceArtifactBundle::from_root(reference_root)?;
        let case = bundle.case_by_id(case_id)?;
        let metadata = case.load_prepared_metadata()?;
        let tokens = case.load_final_tokens()?;
        self.stage1.prepare_decode(&tokens, metadata.ref_rms)
    }

    pub fn decode_stage1_raw_from_reference_case(
        &self,
        reference_root: impl AsRef<std::path::Path>,
        case_id: &str,
    ) -> Result<DecodedAudio> {
        let bundle = ReferenceArtifactBundle::from_root(reference_root)?;
        let case = bundle.case_by_id(case_id)?;
        let metadata = case.load_prepared_metadata()?;
        let tokens = case.load_generated_tokens()?;
        self.stage1.decode_raw(&tokens, metadata.ref_rms)
    }

    pub fn decode_stage1_final_from_reference_case(
        &self,
        reference_root: impl AsRef<std::path::Path>,
        case_id: &str,
    ) -> Result<DecodedAudio> {
        let bundle = ReferenceArtifactBundle::from_root(reference_root)?;
        let case = bundle.case_by_id(case_id)?;
        let metadata = case.load_prepared_metadata()?;
        let tokens = case.load_generated_tokens()?;
        self.stage1
            .decode_final(&tokens, metadata.ref_rms, metadata.postprocess_output)
    }

    pub fn debug_stage1_from_reference_case(
        &self,
        reference_root: impl AsRef<std::path::Path>,
        case_id: &str,
    ) -> Result<Stage1DebugRun> {
        let bundle = ReferenceArtifactBundle::from_root(reference_root)?;
        let case = bundle.case_by_id(case_id)?;
        let metadata = case.load_prepared_metadata()?;
        let tokens = case.load_generated_tokens()?;
        let reference = case.load_stage1_debug_capture()?;
        let actual =
            self.stage1
                .debug_decode(&tokens, metadata.ref_rms, metadata.postprocess_output)?;
        let tensor_metrics =
            crate::stage1_decoder::stage1_tensor_metrics(&actual.tensors, &reference.tensors)?;
        let raw_audio_metrics = actual.raw_audio.parity_metrics(&reference.raw_audio)?;
        let final_audio_metrics = actual.final_audio.parity_metrics(&reference.final_audio)?;
        Ok(Stage1DebugRun {
            tensor_metrics,
            raw_audio_metrics,
            final_audio_metrics,
            debug_capture: actual,
        })
    }

    pub fn runtime_artifacts(&self) -> &RuntimeArtifacts {
        &self.runtime_artifacts
    }

    pub fn frontend(&self) -> &Frontend {
        &self.frontend
    }

    pub fn stage0(&self) -> &Stage0RuntimePlan {
        &self.stage0
    }

    pub fn stage1(&self) -> &Stage1RuntimePlan {
        &self.stage1
    }

    fn materialize_request(&self, request: &GenerationRequest) -> Result<GenerationRequest> {
        let batch_size = request.texts.len();
        let mut request = request.clone();
        request.ref_audios = normalize_option_ref_audio(&request.ref_audios, batch_size)?;
        request.ref_texts = normalize_option_strings(&request.ref_texts, batch_size, "ref_texts")?;
        request.voice_clone_prompts =
            normalize_option_prompts(&request.voice_clone_prompts, batch_size)?;

        for index in 0..batch_size {
            if request.voice_clone_prompts[index].is_some() {
                continue;
            }
            if let Some(ref_audio) = request.ref_audios[index].clone() {
                let prompt = self.create_voice_clone_prompt_from_audio(
                    &ref_audio,
                    request.ref_texts[index].as_deref(),
                    request.generation_config.preprocess_prompt,
                    request.asr_model.as_deref(),
                )?;
                request.voice_clone_prompts[index] = Some(prompt);
                request.ref_texts[index] = None;
            }
        }
        request.ref_audios = vec![None; batch_size];
        Ok(request)
    }

    fn materialize_device_request(
        &self,
        request: &GenerationRequest,
    ) -> Result<MaterializedDeviceRequest> {
        let batch_size = request.texts.len();
        let mut request = request.clone();
        request.ref_audios = normalize_option_ref_audio(&request.ref_audios, batch_size)?;
        request.ref_texts = normalize_option_strings(&request.ref_texts, batch_size, "ref_texts")?;
        request.voice_clone_prompts =
            normalize_option_prompts(&request.voice_clone_prompts, batch_size)?;

        let mut device_voice_clone_prompts = Vec::with_capacity(batch_size);
        for index in 0..batch_size {
            if let Some(prompt) = request.voice_clone_prompts[index].clone() {
                device_voice_clone_prompts.push(Some(DeviceVoiceClonePrompt {
                    ref_audio_tokens: prompt.ref_audio_tokens.to_candle(self.stage0.device())?,
                    ref_text: prompt.ref_text,
                    ref_rms: prompt.ref_rms,
                }));
                continue;
            }
            if let Some(ref_audio) = request.ref_audios[index].clone() {
                device_voice_clone_prompts.push(Some(
                    self.create_device_voice_clone_prompt_from_audio(
                        &ref_audio,
                        request.ref_texts[index].as_deref(),
                        request.generation_config.preprocess_prompt,
                        request.asr_model.as_deref(),
                    )?,
                ));
                continue;
            }
            device_voice_clone_prompts.push(None);
        }
        request.ref_audios = vec![None; batch_size];
        Ok(MaterializedDeviceRequest {
            request,
            device_voice_clone_prompts,
        })
    }

    fn create_device_voice_clone_prompt_from_audio(
        &self,
        ref_audio: &ReferenceAudioInput,
        ref_text: Option<&str>,
        preprocess_prompt: bool,
        asr_model: Option<&str>,
    ) -> Result<DeviceVoiceClonePrompt> {
        let processor = ReferenceAudioProcessor::new(
            self.runtime_artifacts.contracts().sample_rate,
            self.runtime_artifacts.contracts().hop_length,
        );
        let prepared = processor.prepare_prompt_audio(ref_audio, ref_text, preprocess_prompt)?;
        let resolved_ref_text = match prepared.ref_text {
            Some(ref_text) => ref_text,
            None => {
                let resolved_ref_text = self.transcribe_waveform(
                    &WaveformInput::mono(
                        prepared.waveform.clone(),
                        self.runtime_artifacts.contracts().sample_rate,
                    ),
                    asr_model,
                )?;
                if preprocess_prompt {
                    add_punctuation(&resolved_ref_text)
                } else {
                    resolved_ref_text
                }
            }
        };
        let ref_audio_tokens = self.audio_tokenizer.encode_waveform_device(
            &prepared.waveform,
            self.runtime_artifacts.contracts().sample_rate,
        )?;

        Ok(DeviceVoiceClonePrompt {
            ref_audio_tokens,
            ref_text: resolved_ref_text,
            ref_rms: prepared.ref_rms,
        })
    }

    fn generate_tokens_from_task(
        &self,
        task: &crate::contracts::GenerationTask,
    ) -> Result<Vec<GeneratedTokens>> {
        let mut results = vec![None; task.batch_size()];
        let (short_idx, long_idx) = task.get_indices(self.frontend.frame_rate());
        if !short_idx.is_empty() {
            let short_task = task.slice_task(&short_idx);
            let short_results = self.generate_iterative_task(&short_task)?;
            for (slot, generated) in short_idx.into_iter().zip(short_results) {
                results[slot] = Some(GeneratedTokens::Single(generated));
            }
        }
        if !long_idx.is_empty() {
            let long_task = task.slice_task(&long_idx);
            let long_results = self.generate_chunked_task(&long_task)?;
            for (slot, generated) in long_idx.into_iter().zip(long_results) {
                results[slot] = Some(GeneratedTokens::Chunked(generated));
            }
        }
        results
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| {
                    crate::error::OmniVoiceError::InvalidData(
                        "live generation did not produce a result for one of the items".to_string(),
                    )
                })
            })
            .collect()
    }

    fn generate_audio_from_device_task(
        &self,
        task: &DeviceGenerationTask,
    ) -> Result<Vec<DecodedAudio>> {
        let mut results = vec![None; task.batch_size()];
        let (short_idx, long_idx) = task.get_indices(self.frontend.frame_rate());
        if !short_idx.is_empty() {
            let short_task = task.slice_task(&short_idx);
            let short_results = self.generate_iterative_task_device(&short_task)?;
            for (slot, generated) in short_idx.into_iter().zip(short_results) {
                results[slot] = Some(GeneratedTensorTokens::Single(generated));
            }
        }
        if !long_idx.is_empty() {
            let long_task = task.slice_task(&long_idx);
            let long_results = self.generate_chunked_task_device(&long_task)?;
            for (slot, generated) in long_idx.into_iter().zip(long_results) {
                results[slot] = Some(GeneratedTensorTokens::Chunked(generated));
            }
        }
        results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                let generated = match result.ok_or_else(|| {
                    crate::error::OmniVoiceError::InvalidData(
                        "live generation did not produce a result for one of the items".to_string(),
                    )
                })? {
                    GeneratedTensorTokens::Single(tokens) => {
                        GeneratedTokens::Single(tensor_to_i64_tensor2(&tokens)?)
                    }
                    GeneratedTensorTokens::Chunked(chunks) => GeneratedTokens::Chunked(
                        chunks
                            .iter()
                            .map(tensor_to_i64_tensor2)
                            .collect::<Result<Vec<_>>>()?,
                    ),
                };
                self.stage1.decode_final(
                    &generated,
                    task.ref_rms[index],
                    task.generation_config.postprocess_output,
                )
            })
            .collect()
    }

    fn generate_iterative_task(
        &self,
        task: &crate::contracts::GenerationTask,
    ) -> Result<Vec<crate::contracts::I64Tensor2>> {
        let mut prepared = Vec::with_capacity(task.batch_size());
        let mut cond_lens = Vec::with_capacity(task.batch_size());
        for index in 0..task.batch_size() {
            let prompt = self.frontend.prepare_prompt(task, index)?;
            cond_lens.push(prompt.total_length);
            prepared.push(prompt);
        }
        let batched = pack_cfg_batch(&prepared, task.target_lens())?;
        let batch = self
            .stage0
            .prepare_batch(&batched, &cond_lens, task.target_lens())?;
        let generation = self.stage0.generate_deterministic(
            &batch,
            &Stage0DeterministicConfig {
                num_step: task.generation_config.num_step,
                guidance_scale: task.generation_config.guidance_scale,
                t_shift: task.generation_config.t_shift,
                layer_penalty_factor: task.generation_config.layer_penalty_factor,
                position_temperature: task.generation_config.position_temperature,
                class_temperature: task.generation_config.class_temperature,
                capture_steps: Vec::new(),
                capture_layers: Vec::new(),
                capture_final_hidden: false,
            },
            &[],
        )?;
        Ok(generation.tokens)
    }

    fn generate_iterative_task_device(&self, task: &DeviceGenerationTask) -> Result<Vec<Tensor>> {
        let mut prepared = Vec::with_capacity(task.batch_size());
        for index in 0..task.batch_size() {
            let prompt = self
                .frontend
                .prepare_prompt_device(task, index, self.stage0.device())?;
            prepared.push(prompt);
        }
        let batch = pack_cfg_batch_device(
            &prepared,
            task.target_lens(),
            self.stage0.config().num_audio_codebook,
            self.stage0.device(),
            self.stage0.runtime_dtype(),
        )?;
        self.stage0.generate_deterministic_device(
            &batch,
            &Stage0DeterministicConfig {
                num_step: task.generation_config.num_step,
                guidance_scale: task.generation_config.guidance_scale,
                t_shift: task.generation_config.t_shift,
                layer_penalty_factor: task.generation_config.layer_penalty_factor,
                position_temperature: task.generation_config.position_temperature,
                class_temperature: task.generation_config.class_temperature,
                capture_steps: Vec::new(),
                capture_layers: Vec::new(),
                capture_final_hidden: false,
            },
        )
    }

    fn generate_chunked_task(
        &self,
        task: &crate::contracts::GenerationTask,
    ) -> Result<Vec<Vec<crate::contracts::I64Tensor2>>> {
        let all_chunks = (0..task.batch_size())
            .map(|index| {
                Ok(self.frontend.chunk_text(
                    &task.texts[index],
                    task.target_lens[index],
                    task.generation_config.audio_chunk_duration,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let has_ref = task
            .ref_audio_tokens
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        if has_ref.iter().any(|value| *value) && has_ref.iter().any(|value| !*value) {
            return Err(crate::error::OmniVoiceError::InvalidRequest(
                "chunked inference requires all items to either have or not have reference audio"
                    .to_string(),
            ));
        }
        let max_chunks = all_chunks.iter().map(Vec::len).max().unwrap_or(0);
        let mut chunk_results = vec![Vec::new(); task.batch_size()];

        if has_ref.iter().all(|value| *value) {
            for chunk_index in 0..max_chunks {
                let indices = (0..task.batch_size())
                    .filter(|item_index| chunk_index < all_chunks[*item_index].len())
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    continue;
                }
                let generated = self.run_chunk_batch(
                    task,
                    &indices,
                    indices
                        .iter()
                        .map(|index| all_chunks[*index][chunk_index].clone())
                        .collect(),
                    indices
                        .iter()
                        .map(|index| task.ref_audio_tokens[*index].clone())
                        .collect(),
                    indices
                        .iter()
                        .map(|index| task.ref_texts[*index].clone())
                        .collect(),
                )?;
                for (item_index, generated) in indices.iter().copied().zip(generated) {
                    chunk_results[item_index].push(generated);
                }
            }
        } else {
            let first_indices = (0..task.batch_size())
                .filter(|item_index| !all_chunks[*item_index].is_empty())
                .collect::<Vec<_>>();
            let mut first_chunk_refs = vec![None; task.batch_size()];
            let mut first_chunk_texts = vec![None; task.batch_size()];
            if !first_indices.is_empty() {
                let generated = self.run_chunk_batch(
                    task,
                    &first_indices,
                    first_indices
                        .iter()
                        .map(|index| all_chunks[*index][0].clone())
                        .collect(),
                    vec![None; first_indices.len()],
                    vec![None; first_indices.len()],
                )?;
                for (item_index, generated) in first_indices.iter().copied().zip(generated) {
                    first_chunk_refs[item_index] = Some(generated.clone());
                    first_chunk_texts[item_index] = Some(all_chunks[item_index][0].clone());
                    chunk_results[item_index].push(generated);
                }
            }
            for chunk_index in 1..max_chunks {
                let indices = (0..task.batch_size())
                    .filter(|item_index| chunk_index < all_chunks[*item_index].len())
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    continue;
                }
                let ref_audio_tokens = indices
                    .iter()
                    .map(|index| {
                        Ok(Some(first_chunk_refs[*index].clone().ok_or_else(|| {
                            crate::error::OmniVoiceError::InvalidData(
                                "missing cached first chunk reference tokens".to_string(),
                            )
                        })?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let ref_texts = indices
                    .iter()
                    .map(|index| {
                        Ok(Some(first_chunk_texts[*index].clone().ok_or_else(
                            || {
                                crate::error::OmniVoiceError::InvalidData(
                                    "missing cached first chunk reference text".to_string(),
                                )
                            },
                        )?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let generated = self.run_chunk_batch(
                    task,
                    &indices,
                    indices
                        .iter()
                        .map(|index| all_chunks[*index][chunk_index].clone())
                        .collect(),
                    ref_audio_tokens,
                    ref_texts,
                )?;
                for (item_index, generated) in indices.iter().copied().zip(generated) {
                    chunk_results[item_index].push(generated);
                }
            }
        }
        Ok(chunk_results)
    }

    fn run_chunk_batch(
        &self,
        task: &crate::contracts::GenerationTask,
        indices: &[usize],
        texts: Vec<String>,
        ref_audio_tokens: Vec<Option<crate::contracts::I64Tensor2>>,
        ref_texts: Vec<Option<String>>,
    ) -> Result<Vec<crate::contracts::I64Tensor2>> {
        let target_lens = indices
            .iter()
            .enumerate()
            .map(|(local_index, item_index)| {
                self.frontend.estimate_target_tokens(
                    &texts[local_index],
                    ref_texts[local_index].as_deref(),
                    ref_audio_tokens[local_index]
                        .as_ref()
                        .map(|tokens| tokens.dims().1),
                    task.speed[*item_index],
                )
            })
            .collect::<Vec<_>>();
        let sub_task = crate::contracts::GenerationTask {
            texts,
            target_lens,
            langs: indices
                .iter()
                .map(|index| task.langs[*index].clone())
                .collect(),
            instructs: indices
                .iter()
                .map(|index| task.instructs[*index].clone())
                .collect(),
            ref_texts,
            ref_audio_tokens,
            ref_rms: indices.iter().map(|index| task.ref_rms[*index]).collect(),
            speed: indices.iter().map(|index| task.speed[*index]).collect(),
            generation_config: task.generation_config.clone(),
        };
        self.generate_iterative_task(&sub_task)
    }

    fn generate_chunked_task_device(
        &self,
        task: &DeviceGenerationTask,
    ) -> Result<Vec<Vec<Tensor>>> {
        let all_chunks = (0..task.batch_size())
            .map(|index| {
                Ok(self.frontend.chunk_text(
                    &task.texts[index],
                    task.target_lens[index],
                    task.generation_config.audio_chunk_duration,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let has_ref = task
            .ref_audio_tokens
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        if has_ref.iter().any(|value| *value) && has_ref.iter().any(|value| !*value) {
            return Err(crate::error::OmniVoiceError::InvalidRequest(
                "chunked inference requires all items to either have or not have reference audio"
                    .to_string(),
            ));
        }
        let max_chunks = all_chunks.iter().map(Vec::len).max().unwrap_or(0);
        let mut chunk_results = (0..task.batch_size())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<Tensor>>>();

        if has_ref.iter().all(|value| *value) {
            for chunk_index in 0..max_chunks {
                let indices = (0..task.batch_size())
                    .filter(|item_index| chunk_index < all_chunks[*item_index].len())
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    continue;
                }
                let generated = self.run_chunk_batch_device(
                    task,
                    &indices,
                    indices
                        .iter()
                        .map(|index| all_chunks[*index][chunk_index].clone())
                        .collect(),
                    indices
                        .iter()
                        .map(|index| task.ref_audio_tokens[*index].clone())
                        .collect(),
                    indices
                        .iter()
                        .map(|index| task.ref_texts[*index].clone())
                        .collect(),
                )?;
                for (item_index, generated) in indices.iter().copied().zip(generated) {
                    chunk_results[item_index].push(generated);
                }
            }
        } else {
            let first_indices = (0..task.batch_size())
                .filter(|item_index| !all_chunks[*item_index].is_empty())
                .collect::<Vec<_>>();
            let mut first_chunk_refs = vec![None; task.batch_size()];
            let mut first_chunk_texts = vec![None; task.batch_size()];
            if !first_indices.is_empty() {
                let generated = self.run_chunk_batch_device(
                    task,
                    &first_indices,
                    first_indices
                        .iter()
                        .map(|index| all_chunks[*index][0].clone())
                        .collect(),
                    vec![None; first_indices.len()],
                    vec![None; first_indices.len()],
                )?;
                for (item_index, generated) in first_indices.iter().copied().zip(generated) {
                    first_chunk_refs[item_index] = Some(generated.clone());
                    first_chunk_texts[item_index] = Some(all_chunks[item_index][0].clone());
                    chunk_results[item_index].push(generated);
                }
            }
            for chunk_index in 1..max_chunks {
                let indices = (0..task.batch_size())
                    .filter(|item_index| chunk_index < all_chunks[*item_index].len())
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    continue;
                }
                let ref_audio_tokens = indices
                    .iter()
                    .map(|index| {
                        Ok(Some(first_chunk_refs[*index].clone().ok_or_else(|| {
                            crate::error::OmniVoiceError::InvalidData(
                                "missing cached first chunk reference tokens".to_string(),
                            )
                        })?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let ref_texts = indices
                    .iter()
                    .map(|index| {
                        Ok(Some(first_chunk_texts[*index].clone().ok_or_else(
                            || {
                                crate::error::OmniVoiceError::InvalidData(
                                    "missing cached first chunk reference text".to_string(),
                                )
                            },
                        )?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let generated = self.run_chunk_batch_device(
                    task,
                    &indices,
                    indices
                        .iter()
                        .map(|index| all_chunks[*index][chunk_index].clone())
                        .collect(),
                    ref_audio_tokens,
                    ref_texts,
                )?;
                for (item_index, generated) in indices.iter().copied().zip(generated) {
                    chunk_results[item_index].push(generated);
                }
            }
        }
        Ok(chunk_results)
    }

    fn run_chunk_batch_device(
        &self,
        task: &DeviceGenerationTask,
        indices: &[usize],
        texts: Vec<String>,
        ref_audio_tokens: Vec<Option<Tensor>>,
        ref_texts: Vec<Option<String>>,
    ) -> Result<Vec<Tensor>> {
        let target_lens = indices
            .iter()
            .enumerate()
            .map(|(local_index, item_index)| {
                let num_ref_audio_tokens = ref_audio_tokens[local_index]
                    .as_ref()
                    .map(|tokens| tokens.dims2().map(|(_, steps)| steps))
                    .transpose()?;
                Ok(self.frontend.estimate_target_tokens(
                    &texts[local_index],
                    ref_texts[local_index].as_deref(),
                    num_ref_audio_tokens,
                    task.speed[*item_index],
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let sub_task = DeviceGenerationTask {
            texts,
            target_lens,
            langs: indices
                .iter()
                .map(|index| task.langs[*index].clone())
                .collect(),
            instructs: indices
                .iter()
                .map(|index| task.instructs[*index].clone())
                .collect(),
            ref_texts,
            ref_audio_tokens,
            ref_rms: indices.iter().map(|index| task.ref_rms[*index]).collect(),
            speed: indices.iter().map(|index| task.speed[*index]).collect(),
            generation_config: task.generation_config.clone(),
        };
        self.generate_iterative_task_device(&sub_task)
    }

    fn generate_stage0_from_prepared_prompts(
        &self,
        prepared: &PreparedPromptSequence,
    ) -> Result<GeneratedTokens> {
        match prepared {
            PreparedPromptSequence::Single(prompt) => {
                let batch = self.prepare_stage0_prompt(prompt)?;
                let generation = self.stage0.generate_deterministic(
                    &batch,
                    &Stage0DeterministicConfig::default(),
                    &[],
                )?;
                Ok(GeneratedTokens::Single(
                    generation.tokens.into_iter().next().ok_or_else(|| {
                        crate::error::OmniVoiceError::InvalidData(
                            "stage0 generation did not return any token tensors".to_string(),
                        )
                    })?,
                ))
            }
            PreparedPromptSequence::Chunked(chunked) => {
                let mut chunks = Vec::with_capacity(chunked.prompts.len());
                for prompt in &chunked.prompts {
                    let batch = self.prepare_stage0_prompt(prompt)?;
                    let generation = self.stage0.generate_deterministic(
                        &batch,
                        &Stage0DeterministicConfig::default(),
                        &[],
                    )?;
                    chunks.push(generation.tokens.into_iter().next().ok_or_else(|| {
                        crate::error::OmniVoiceError::InvalidData(
                            "chunked stage0 generation did not return a token tensor".to_string(),
                        )
                    })?);
                }
                Ok(GeneratedTokens::Chunked(chunks))
            }
        }
    }

    fn prepare_stage0_prompt(&self, prompt: &PreparedPrompt) -> Result<PreparedInferenceBatch> {
        let cond_lens = vec![prompt.total_length];
        let target_lens = vec![prompt.target_length];
        let batched = pack_cfg_batch(std::slice::from_ref(prompt), &target_lens)?;
        self.stage0
            .prepare_batch(&batched, &cond_lens, &target_lens)
    }

    fn estimate_generation_usage(
        &self,
        request: &GenerationRequest,
        task: &DeviceGenerationTask,
    ) -> Result<Vec<GenerationUsage>> {
        let mut usage = Vec::with_capacity(task.batch_size());
        for index in 0..task.batch_size() {
            let mut input_tokens = self.frontend.count_text_tokens(&request.texts[index])?;
            if let Some(language) = request
                .languages
                .get(index)
                .and_then(|value| value.as_deref())
            {
                input_tokens += self.frontend.count_text_tokens(language)?;
            }
            if let Some(instruct) = task.instructs.get(index).and_then(|value| value.as_deref()) {
                input_tokens += self.frontend.count_text_tokens(instruct)?;
            }
            if let Some(prompt) = request
                .voice_clone_prompts
                .get(index)
                .and_then(|value| value.as_ref())
            {
                input_tokens += self.frontend.count_text_tokens(&prompt.ref_text)?;
            } else if let Some(ref_text) = request
                .ref_texts
                .get(index)
                .and_then(|value| value.as_deref())
            {
                input_tokens += self.frontend.count_text_tokens(ref_text)?;
            }

            usage.push(GenerationUsage::new(input_tokens, task.target_lens[index]));
        }
        Ok(usage)
    }
}

fn pack_cfg_batch_device(
    prepared: &[PreparedPromptDevice],
    target_lens: &[usize],
    num_audio_codebook: usize,
    device: &Device,
    runtime_dtype: DType,
) -> Result<PreparedInferenceBatch> {
    if prepared.is_empty() {
        return Err(crate::error::OmniVoiceError::InvalidRequest(
            "prepared prompts cannot be empty".to_string(),
        ));
    }
    if prepared.len() != target_lens.len() {
        return Err(crate::error::OmniVoiceError::InvalidRequest(format!(
            "prepared prompt count {} does not match target lens {}",
            prepared.len(),
            target_lens.len()
        )));
    }

    let max_c_len = prepared
        .iter()
        .map(|prompt| prompt.total_length)
        .max()
        .unwrap_or_default();
    let max_target_len = target_lens.iter().copied().max().unwrap_or(0);
    let audio_mask_id = prepared[0].audio_mask_id;

    let mut cond_inputs = Vec::with_capacity(prepared.len());
    let mut cond_audio_masks = Vec::with_capacity(prepared.len());
    let mut cond_attention_masks = Vec::with_capacity(prepared.len());
    let mut uncond_inputs = Vec::with_capacity(prepared.len());
    let mut uncond_audio_masks = Vec::with_capacity(prepared.len());
    let mut uncond_attention_masks = Vec::with_capacity(prepared.len());
    let mut cond_lens = Vec::with_capacity(prepared.len());

    for (prompt, target_len) in prepared.iter().zip(target_lens.iter().copied()) {
        cond_lens.push(prompt.total_length);
        cond_inputs.push(pad_input_ids_tensor(
            &prompt.input_ids,
            max_c_len,
            audio_mask_id,
            num_audio_codebook,
            device,
        )?);
        cond_audio_masks.push(pad_mask_tensor(&prompt.audio_mask, max_c_len, device)?);
        cond_attention_masks.push(build_attention_mask_tensor(
            prompt.total_length,
            max_c_len,
            None,
            device,
        )?);

        let uncond_source = prompt.input_ids.narrow(
            candle_core::D::Minus1,
            prompt.total_length - target_len,
            target_len,
        )?;
        let uncond_mask_source = prompt.audio_mask.narrow(
            candle_core::D::Minus1,
            prompt.total_length - target_len,
            target_len,
        )?;
        uncond_inputs.push(pad_input_ids_tensor(
            &uncond_source,
            max_c_len,
            audio_mask_id,
            num_audio_codebook,
            device,
        )?);
        uncond_audio_masks.push(pad_mask_tensor(&uncond_mask_source, max_c_len, device)?);
        uncond_attention_masks.push(build_attention_mask_tensor(
            target_len,
            max_c_len,
            Some(target_len),
            device,
        )?);
    }

    let input_tensor_refs = cond_inputs
        .iter()
        .chain(uncond_inputs.iter())
        .collect::<Vec<_>>();
    let audio_mask_refs = cond_audio_masks
        .iter()
        .chain(uncond_audio_masks.iter())
        .collect::<Vec<_>>();
    let attention_mask_refs = cond_attention_masks
        .iter()
        .chain(uncond_attention_masks.iter())
        .collect::<Vec<_>>();

    Ok(PreparedInferenceBatch {
        input_ids: Tensor::cat(&input_tensor_refs, 0)?,
        audio_mask: Tensor::cat(&audio_mask_refs, 0)?,
        attention_mask: Tensor::cat(&attention_mask_refs, 0)?,
        tokens_init: Tensor::full(
            audio_mask_id,
            (prepared.len(), num_audio_codebook, max_target_len),
            device,
        )?,
        target_lens: target_lens.to_vec(),
        cond_lens,
        runtime_dtype,
    })
}

fn pad_input_ids_tensor(
    input_ids: &Tensor,
    max_c_len: usize,
    audio_mask_id: i64,
    num_audio_codebook: usize,
    device: &Device,
) -> Result<Tensor> {
    let (_, _, current_len) = input_ids.dims3()?;
    if current_len >= max_c_len {
        return Ok(input_ids.clone());
    }
    let padding = Tensor::full(
        audio_mask_id,
        (1, num_audio_codebook, max_c_len - current_len),
        device,
    )?;
    Tensor::cat(&[input_ids, &padding], 2).map_err(Into::into)
}

fn pad_mask_tensor(mask: &Tensor, max_c_len: usize, device: &Device) -> Result<Tensor> {
    let (_, current_len) = mask.dims2()?;
    if current_len >= max_c_len {
        return Ok(mask.clone());
    }
    let padding = Tensor::zeros((1, max_c_len - current_len), DType::U8, device)?;
    Tensor::cat(&[mask, &padding], 1).map_err(Into::into)
}

fn build_attention_mask_tensor(
    active_len: usize,
    max_c_len: usize,
    diagonal_from: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let mut values = vec![0u8; max_c_len * max_c_len];
    for row in 0..active_len {
        let row_offset = row * max_c_len;
        for column in 0..active_len {
            values[row_offset + column] = 1;
        }
    }
    if let Some(start) = diagonal_from {
        for index in start..max_c_len {
            values[index * max_c_len + index] = 1;
        }
    }
    Ok(Tensor::from_vec(
        values,
        (1, 1, max_c_len, max_c_len),
        device,
    )?)
}

fn normalize_option_strings(
    values: &[Option<String>],
    batch_size: usize,
    name: &str,
) -> Result<Vec<Option<String>>> {
    if values.len() == batch_size {
        Ok(values.to_vec())
    } else if values.len() == 1 {
        Ok(vec![values[0].clone(); batch_size])
    } else {
        Err(crate::error::OmniVoiceError::InvalidRequest(format!(
            "{name} should contain either 1 or {batch_size} items, got {}",
            values.len()
        )))
    }
}

fn normalize_option_ref_audio(
    values: &[Option<ReferenceAudioInput>],
    batch_size: usize,
) -> Result<Vec<Option<ReferenceAudioInput>>> {
    if values.len() == batch_size {
        Ok(values.to_vec())
    } else if values.len() == 1 {
        Ok(vec![values[0].clone(); batch_size])
    } else {
        Err(crate::error::OmniVoiceError::InvalidRequest(format!(
            "ref_audios should contain either 1 or {batch_size} items, got {}",
            values.len()
        )))
    }
}

fn normalize_option_prompts(
    values: &[Option<VoiceClonePrompt>],
    batch_size: usize,
) -> Result<Vec<Option<VoiceClonePrompt>>> {
    if values.len() == batch_size {
        Ok(values.to_vec())
    } else if values.len() == 1 {
        Ok(vec![values[0].clone(); batch_size])
    } else {
        Err(crate::error::OmniVoiceError::InvalidRequest(format!(
            "voice_clone_prompts should contain either 1 or {batch_size} items, got {}",
            values.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;
    use crate::{
        artifacts::ReferenceArtifactBundle,
        runtime::{DTypeSpec, DeviceSpec},
        stage0_loop::pack_cfg_batch,
        stage0_model::Stage0RuntimePlan,
    };

    fn repo_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn model_root() -> std::path::PathBuf {
        repo_root().join("model")
    }

    fn deterministic_reference_root() -> std::path::PathBuf {
        repo_root()
            .join("artifacts")
            .join("python_reference_stage7_cuda_f32_dense")
    }

    fn cpu_frontend_and_stage0() -> (Frontend, Stage0RuntimePlan) {
        let options = RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cpu)
            .with_dtype(DTypeSpec::F32)
            .with_seed(1234);
        let runtime = RuntimeArtifacts::from_model_root(model_root()).unwrap();
        let frontend = Frontend::from_runtime_artifacts(&runtime).unwrap();
        let stage0 =
            Stage0RuntimePlan::from_runtime_artifacts_with_device(options, &runtime, Device::Cpu)
                .unwrap();
        (frontend, stage0)
    }

    fn assert_batch_matches_canonical(request: &GenerationRequest) {
        let (frontend, stage0) = cpu_frontend_and_stage0();
        let task = frontend.build_task(request).unwrap();
        let mut prepared = Vec::with_capacity(task.batch_size());
        let mut cond_lens = Vec::with_capacity(task.batch_size());
        for index in 0..task.batch_size() {
            let prompt = frontend.prepare_prompt(&task, index).unwrap();
            cond_lens.push(prompt.total_length);
            prepared.push(prompt);
        }
        let canonical_inputs = pack_cfg_batch(&prepared, task.target_lens()).unwrap();
        let canonical = stage0
            .prepare_batch(&canonical_inputs, &cond_lens, task.target_lens())
            .unwrap();

        let device_task = frontend
            .build_task_with_device_prompts(request, &vec![None; request.texts.len()])
            .unwrap();
        let mut prepared_device = Vec::with_capacity(device_task.batch_size());
        for index in 0..device_task.batch_size() {
            prepared_device.push(
                frontend
                    .prepare_prompt_device(&device_task, index, stage0.device())
                    .unwrap(),
            );
        }
        let device_prepared = pack_cfg_batch_device(
            &prepared_device,
            device_task.target_lens(),
            stage0.config().num_audio_codebook,
            stage0.device(),
            DType::F32,
        )
        .unwrap();

        assert_eq!(device_prepared.target_lens, canonical.target_lens);
        assert_eq!(device_prepared.cond_lens, canonical.cond_lens);
        assert_eq!(device_prepared.runtime_dtype, canonical.runtime_dtype);
        assert_eq!(device_prepared.input_ids.dims(), canonical.input_ids.dims());
        assert_eq!(
            device_prepared.audio_mask.dims(),
            canonical.audio_mask.dims()
        );
        assert_eq!(
            device_prepared.attention_mask.dims(),
            canonical.attention_mask.dims()
        );
        assert_eq!(
            device_prepared.tokens_init.dims(),
            canonical.tokens_init.dims()
        );
        assert_eq!(
            device_prepared
                .input_ids
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<i64>()
                .unwrap(),
            canonical
                .input_ids
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<i64>()
                .unwrap(),
        );
        assert_eq!(
            device_prepared
                .audio_mask
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
            canonical
                .audio_mask
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
        );
        assert_eq!(
            device_prepared
                .attention_mask
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
            canonical
                .attention_mask
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
        );
        assert_eq!(
            device_prepared
                .tokens_init
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<i64>()
                .unwrap(),
            canonical
                .tokens_init
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<i64>()
                .unwrap(),
        );
    }

    #[test]
    fn device_batch_preparation_matches_canonical_for_design_prompt() {
        let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
        let request = bundle
            .case_by_id("det_design_en_british")
            .unwrap()
            .build_generation_request()
            .unwrap();
        assert_batch_matches_canonical(&request);
    }

    #[test]
    fn device_batch_preparation_matches_canonical_for_mixed_short_and_long_batch() {
        let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
        let auto_case = bundle.case_by_id("det_auto_en_short").unwrap();
        let auto_request = auto_case.build_generation_request().unwrap();
        let mut long_request = auto_case.build_generation_request().unwrap();
        long_request.durations = vec![Some(31.0)];

        let mut request = GenerationRequest::new_text_only(auto_request.texts[0].clone());
        request.texts = vec![auto_request.texts[0].clone(), long_request.texts[0].clone()];
        request.languages = vec![
            auto_request.languages[0].clone(),
            long_request.languages[0].clone(),
        ];
        request.instructs = vec![
            auto_request.instructs[0].clone(),
            long_request.instructs[0].clone(),
        ];
        request.ref_texts = vec![None, None];
        request.ref_audios = vec![None, None];
        request.voice_clone_prompts = vec![None, None];
        request.speeds = vec![auto_request.speeds[0], long_request.speeds[0]];
        request.durations = vec![auto_request.durations[0], long_request.durations[0]];
        request.generation_config = auto_request.generation_config.clone();

        assert_batch_matches_canonical(&request);
    }
}
