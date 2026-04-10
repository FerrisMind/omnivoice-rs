#![cfg(all(feature = "metal", target_os = "macos"))]

mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::{GenerationRequest, ReferenceAudioInput, VoiceClonePrompt},
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{
    assert_audio_matches_reference_with_frame_tolerance, deterministic_reference_root,
    live_oracle_clone_prompt, model_root, ref_audio_path, reference_root,
};

fn metal_f32_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Metal)
            .with_dtype(DTypeSpec::F32)
            .with_seed(1234),
    )
    .unwrap()
}

#[test]
fn live_auto_request_matches_reference_audio_metal() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    let metrics = actual[0].parity_metrics(&expected).unwrap();
    assert_eq!(metrics.sample_rate, 24_000);
    assert!(metrics.mae < 2.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 3.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-3, "{metrics:?}");
}

#[test]
fn live_design_request_matches_reference_audio_metal() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_design_en_british").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 480, 2.0e-3, 6.0e-3, 0.35,
    );
}

#[test]
fn live_clone_prompt_from_raw_audio_matches_reference_tokens_metal() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let (reference_tokens, reference_text) = live_oracle_clone_prompt();
    let prompt = metal_f32_pipeline()
        .create_voice_clone_prompt_from_audio(
            &ReferenceAudioInput::from_path(ref_audio_path().display().to_string()),
            Some("State-of-the-art text-to-speech model for 600+ languages, supporting"),
            true,
            None,
        )
        .unwrap();
    assert_eq!(prompt.ref_audio_tokens.dims(), reference_tokens.dims());
    assert!(prompt
        .ref_audio_tokens
        .data
        .iter()
        .all(|token| (0..=1023).contains(token)));
    assert_eq!(prompt.ref_text, reference_text);
}

#[test]
fn live_clone_request_from_raw_audio_matches_reference_audio_metal() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let mut request = case.build_generation_request().unwrap();
    request.ref_audios = vec![Some(ReferenceAudioInput::from_path(
        ref_audio_path().display().to_string(),
    ))];
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn live_clone_request_without_ref_text_uses_whisper_asr_metal() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let definition = case.load_case_definition().unwrap();
    let mut request = GenerationRequest::new_text_only(definition.request.texts[0].clone());
    request.languages = definition.request.languages.clone();
    request.ref_audios = vec![Some(ReferenceAudioInput::from_path(
        ref_audio_path().display().to_string(),
    ))];
    request.ref_texts = vec![None];
    request.generation_config = definition.request.generation_config.clone();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn live_clone_request_from_prebuilt_prompt_matches_reference_audio_metal() {
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
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn live_long_form_auto_matches_reference_audio_metal() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_long_chunked").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(actual[0].frame_count() > 240_000);
    assert!(!actual[0].samples.is_empty());
}
