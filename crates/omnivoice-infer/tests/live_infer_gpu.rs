#![cfg(feature = "cuda")]

mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::{
        DecodedAudio, GenerationRequest, I64Tensor2, ReferenceAudioInput, VoiceClonePrompt,
    },
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{deterministic_reference_root, model_root, ref_audio_path, reference_root};

fn cuda_f32_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F32)
            .with_seed(1234),
    )
    .unwrap()
}

fn live_oracle_clone_prompt() -> (I64Tensor2, String) {
    let value: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            support::repo_root()
                .join("artifacts")
                .join("live_oracles")
                .join("clone_prompt.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let dims = value["dims"].as_array().unwrap();
    let rows = dims[0].as_u64().unwrap() as usize;
    let cols = dims[1].as_u64().unwrap() as usize;
    let mut flat = Vec::with_capacity(rows * cols);
    for row in value["tokens"].as_array().unwrap() {
        for token in row.as_array().unwrap() {
            flat.push(token.as_i64().unwrap());
        }
    }
    (
        I64Tensor2::new((rows, cols), flat).unwrap(),
        value["ref_text"].as_str().unwrap().to_string(),
    )
}

#[test]
fn live_auto_request_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let pipeline = cuda_f32_pipeline();
    let actual = pipeline.generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_eq!(actual.len(), 1);
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 480, 5.0e-4, 8.0e-4, 0.05,
    );
}

#[test]
fn live_design_request_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_design_en_british").unwrap();
    let request = case.build_generation_request().unwrap();
    let pipeline = cuda_f32_pipeline();
    let actual = pipeline.generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 480, 2.0e-3, 6.0e-3, 0.35,
    );
}

#[test]
fn live_clone_prompt_from_raw_audio_matches_reference_tokens() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let (reference_tokens, reference_text) = live_oracle_clone_prompt();
    let prompt = cuda_f32_pipeline()
        .create_voice_clone_prompt_from_audio(
            &ReferenceAudioInput::from_path(ref_audio_path()),
            Some("State-of-the-art text-to-speech model for 600+ languages, supporting"),
            true,
            None,
        )
        .unwrap();
    assert_eq!(prompt.ref_audio_tokens.dims(), reference_tokens.dims());
    assert_token_domain(&prompt.ref_audio_tokens.data);
    assert_eq!(prompt.ref_text, reference_text);
}

#[test]
fn live_clone_request_from_raw_audio_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let mut request = case.build_generation_request().unwrap();
    request.ref_audios = vec![Some(ReferenceAudioInput::from_path(ref_audio_path()))];
    let actual = cuda_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn live_clone_request_without_ref_text_uses_whisper_asr() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let definition = case.load_case_definition().unwrap();
    let mut request = GenerationRequest::new_text_only(definition.request.texts[0].clone());
    request.languages = definition.request.languages.clone();
    request.ref_audios = vec![Some(ReferenceAudioInput::from_path(ref_audio_path()))];
    request.ref_texts = vec![None];
    request.generation_config = definition.request.generation_config.clone();
    let actual = cuda_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn live_clone_request_from_prebuilt_prompt_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let mut request = case.build_generation_request().unwrap();
    let (reference_tokens, reference_text) = live_oracle_clone_prompt();
    request.ref_audios = vec![None];
    request.ref_texts = vec![None];
    request.voice_clone_prompts = vec![Some(VoiceClonePrompt {
        ref_audio_tokens: reference_tokens,
        ref_text: reference_text,
        ref_rms: Some(0.1),
    })];
    let actual = cuda_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn live_long_form_auto_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_long_chunked").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = cuda_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(actual[0].frame_count() > 240_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn live_batch_request_preserves_order() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let auto_case = bundle.case_by_id("det_auto_en_short").unwrap();
    let auto_request = auto_case.build_generation_request().unwrap();
    let mut request = GenerationRequest::new_text_only(auto_request.texts[0].clone());
    request.texts = vec![auto_request.texts[0].clone(), auto_request.texts[0].clone()];
    request.languages = vec![
        auto_request.languages[0].clone(),
        auto_request.languages[0].clone(),
    ];
    request.instructs = vec![
        auto_request.instructs[0].clone(),
        auto_request.instructs[0].clone(),
    ];
    request.ref_texts = vec![None, None];
    request.ref_audios = vec![None, None];
    request.voice_clone_prompts = vec![None, None];
    request.speeds = vec![auto_request.speeds[0], auto_request.speeds[0]];
    request.durations = vec![auto_request.durations[0], auto_request.durations[0]];
    request.generation_config = auto_request.generation_config.clone();
    let actual = cuda_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert_eq!(actual[1].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
    assert!(!actual[1].samples.is_empty());
}

fn assert_token_domain(tokens: &[i64]) {
    assert!(tokens.iter().all(|token| (0..=1023).contains(token)));
}

fn assert_audio_matches_reference_with_frame_tolerance(
    actual: &DecodedAudio,
    expected: &DecodedAudio,
    max_frame_delta: usize,
    mae_limit: f32,
    rmse_limit: f32,
    max_abs_limit: f32,
) {
    assert_eq!(actual.sample_rate, expected.sample_rate);
    let frame_delta = actual.frame_count().abs_diff(expected.frame_count());
    assert!(
        frame_delta <= max_frame_delta,
        "frame delta {} exceeds {} (actual={}, reference={})",
        frame_delta,
        max_frame_delta,
        actual.frame_count(),
        expected.frame_count()
    );
    let compare_len = actual.frame_count().min(expected.frame_count());
    let actual = DecodedAudio::new(actual.samples[..compare_len].to_vec(), actual.sample_rate);
    let expected = DecodedAudio::new(
        expected.samples[..compare_len].to_vec(),
        expected.sample_rate,
    );
    let metrics = actual.parity_metrics(&expected).unwrap();
    assert!(metrics.mae < mae_limit, "{metrics:?}");
    assert!(metrics.rmse < rmse_limit, "{metrics:?}");
    assert!(metrics.max_abs < max_abs_limit, "{metrics:?}");
}
