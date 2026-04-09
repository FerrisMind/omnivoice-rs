use std::sync::Mutex;

use candle_core::Tensor;

use crate::{
    artifacts::{ReferenceArtifactBundle, RuntimeArtifacts},
    asr::WhisperAsr,
    audio_input::ReferenceAudioProcessor,
    audio_tokenizer::AudioTokenizerRuntimePlan,
    contracts::{
        DecodedAudio, GeneratedTokens, GenerationRequest, PreparedInferenceBatch, PreparedPrompt,
        PreparedPromptSequence, ReferenceAudioInput, VoiceClonePrompt, WaveformInput,
    },
    error::Result,
    frontend::add_punctuation,
    frontend::Frontend,
    reference_prompt::{ReferencePromptBuilder, ReferencePromptOptions},
    runtime::RuntimeOptions,
    stage0_loop::pack_cfg_batch,
    stage0_model::{Stage0DebugRun, Stage0DeterministicConfig, Stage0RuntimePlan},
    stage1_decoder::{PreparedStage1Decode, Stage1DebugRun, Stage1RuntimePlan},
};

#[derive(Clone)]
enum GeneratedTensorTokens {
    Single(Tensor),
    Chunked(Vec<Tensor>),
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
        let audio_tokenizer =
            AudioTokenizerRuntimePlan::from_runtime_artifacts(options.clone(), &runtime_artifacts)?;
        let stage0 =
            Stage0RuntimePlan::from_runtime_artifacts(options.clone(), &runtime_artifacts)?;
        let stage1 =
            Stage1RuntimePlan::from_runtime_artifacts(options.clone(), &runtime_artifacts)?;

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
        let request = self.materialize_request(request)?;
        let task = self.frontend.build_task(&request)?;
        self.generate_audio_from_task(&task)
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

    fn generate_audio_from_task(
        &self,
        task: &crate::contracts::GenerationTask,
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
                match result.ok_or_else(|| {
                    crate::error::OmniVoiceError::InvalidData(
                        "live generation did not produce a result for one of the items".to_string(),
                    )
                })? {
                    GeneratedTensorTokens::Single(tokens) => self.stage1.decode_final_tensor(
                        &tokens,
                        task.ref_rms[index],
                        task.generation_config.postprocess_output,
                    ),
                    GeneratedTensorTokens::Chunked(chunks) => {
                        self.stage1.decode_final_tensor_chunks(
                            &chunks,
                            task.ref_rms[index],
                            task.generation_config.postprocess_output,
                        )
                    }
                }
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

    fn generate_iterative_task_device(
        &self,
        task: &crate::contracts::GenerationTask,
    ) -> Result<Vec<Tensor>> {
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
        task: &crate::contracts::GenerationTask,
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
                    first_chunk_refs[item_index] =
                        Some(crate::stage0_model::tensor_to_i64_tensor2(&generated)?);
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
        task: &crate::contracts::GenerationTask,
        indices: &[usize],
        texts: Vec<String>,
        ref_audio_tokens: Vec<Option<crate::contracts::I64Tensor2>>,
        ref_texts: Vec<Option<String>>,
    ) -> Result<Vec<Tensor>> {
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
