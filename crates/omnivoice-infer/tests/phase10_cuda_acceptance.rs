#![cfg(feature = "cuda")]

mod support;

use candle_core::DType;
use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::{
        DecodedAudio, GenerationRequest, I64Tensor2, ReferenceAudioInput, VoiceClonePrompt,
        WaveformInput,
    },
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
    workspace_phase_marker,
};
use support::{
    assert_audio_matches_reference_with_frame_tolerance, deterministic_reference_root,
    live_oracle_clone_prompt, model_root, ref_audio_path, reference_root,
};

fn cuda_f32_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F32)
            .with_seed(1234),
    )
    .unwrap()
}

fn auto_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Auto)
            .with_dtype(DTypeSpec::Auto)
            .with_seed(1234),
    )
    .unwrap()
}

fn multiline_non_english_text() -> String {
    [
        "第一行 你好世界，我们正在验证多行中文长文本推理。",
        "第二行\t这里包含额外空白和换行，用来覆盖上游兼容的文本归一化。",
        "第三行 这一段继续拉长文本，确保请求稳定进入 chunked inference 路径。",
        "第四行 我们需要确认后续分块和参考提示都保持在合法音频 token 域内。",
    ]
    .join("\n")
}

#[test]
fn phase10_marker_is_current() {
    assert_eq!(workspace_phase_marker(), "omnivoice-phase10");
}

#[test]
fn phase10_cuda_auto_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = cuda_f32_pipeline().generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_eq!(actual.len(), 1);
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 480, 5.0e-4, 8.0e-4, 0.05,
    );
}

#[test]
fn phase10_cuda_design_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_design_en_british").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = cuda_f32_pipeline().generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 480, 2.0e-3, 6.0e-3, 0.35,
    );
}

#[test]
fn phase10_cuda_clone_with_asr_succeeds() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = cuda_f32_pipeline();
    let (reference_tokens, reference_text) = live_oracle_clone_prompt();
    let prompt = pipeline
        .create_voice_clone_prompt_from_audio(
            &ReferenceAudioInput::from_path(ref_audio_path().display().to_string()),
            None,
            true,
            None,
        )
        .unwrap();
    assert_eq!(prompt.ref_audio_tokens.dims(), reference_tokens.dims());
    assert_eq!(
        prompt.ref_text.replace(',', ""),
        reference_text.replace(',', "")
    );
    assert!(prompt
        .ref_audio_tokens
        .data
        .iter()
        .all(|token| (0..=1023).contains(token)));
}

#[test]
fn phase10_cuda_clone_with_prebuilt_prompt_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let pipeline = cuda_f32_pipeline();
    let mut request = case.build_generation_request().unwrap();
    let (reference_tokens, reference_text) = live_oracle_clone_prompt();
    request.ref_audios = vec![None];
    request.ref_texts = vec![None];
    request.voice_clone_prompts = vec![Some(VoiceClonePrompt {
        ref_audio_tokens: reference_tokens,
        ref_text: reference_text,
        ref_rms: Some(0.1),
    })];
    let expected_request = case.build_generation_request().unwrap();
    assert_eq!(
        pipeline.generate_tokens(&request).unwrap(),
        pipeline.generate_tokens(&expected_request).unwrap()
    );
}

#[test]
fn phase10_cuda_invalid_voice_clone_prompt_fails_before_kernel_launch() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let mut tokens = vec![0_i64; 8 * 16];
    tokens[5] = 1024;

    let mut request = GenerationRequest::new_text_only(multiline_non_english_text());
    request.languages = vec![Some("zh".to_string())];
    request.durations = vec![Some(31.0)];
    request.ref_audios = vec![None];
    request.ref_texts = vec![None];
    request.voice_clone_prompts = vec![Some(VoiceClonePrompt {
        ref_audio_tokens: I64Tensor2::new((8, 16), tokens).unwrap(),
        ref_text: "参考文本".to_string(),
        ref_rms: Some(0.1),
    })];

    let error = cuda_f32_pipeline().generate(&request).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("request item 0"), "{message}");
    assert!(message.contains("outside valid range"), "{message}");
}

#[test]
fn phase10_cuda_clone_prompt_creation_requested_f16_succeeds() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F16)
            .with_seed(1234),
    )
    .unwrap();

    let prompt = pipeline
        .create_voice_clone_prompt_from_audio(
            &ReferenceAudioInput::from_path(ref_audio_path().display().to_string()),
            Some("State-of-the-art text-to-speech model for 600+ languages, supporting"),
            true,
            None,
        )
        .unwrap();

    assert!(prompt.ref_audio_tokens.dims().1 > 0);
}

#[test]
fn phase10_cuda_clone_with_user_ref_text_does_not_trim_long_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let reference = DecodedAudio::read_wav(ref_audio_path()).unwrap();
    let repeated = reference.samples.repeat(6);
    let prompt = cuda_f32_pipeline()
        .create_voice_clone_prompt_from_audio(
            &ReferenceAudioInput::Waveform(WaveformInput::mono(repeated, reference.sample_rate)),
            Some("State-of-the-art text-to-speech model for 600+ languages, supporting"),
            true,
            None,
        )
        .unwrap();
    assert!(
        prompt.ref_audio_tokens.dims().1 > 375,
        "reference prompt was unexpectedly trimmed to {} frames",
        prompt.ref_audio_tokens.dims().1
    );
}

#[test]
fn phase10_cuda_long_form_auto_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_long_chunked").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = cuda_f32_pipeline().generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 1_200, 6.0e-3, 3.0e-2, 0.7,
    );
}

#[test]
fn phase10_cuda_batch_request_preserves_order() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let auto_case = bundle.case_by_id("det_auto_en_short").unwrap();
    let auto_request = auto_case.build_generation_request().unwrap();
    let mut long_request = auto_case.build_generation_request().unwrap();
    long_request.durations = vec![Some(31.0)];
    let pipeline = cuda_f32_pipeline();
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
    let actual = pipeline.generate(&request).unwrap();
    let expected_auto = pipeline.generate(&auto_request).unwrap();
    let expected_long = pipeline.generate(&long_request).unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0],
        &expected_auto[0],
        480,
        5.0e-4,
        8.0e-4,
        0.05,
    );
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[1],
        &expected_long[0],
        1_200,
        6.0e-3,
        3.0e-2,
        0.7,
    );
}

#[test]
fn phase10_cuda_mixed_batch_live_audio_matches_token_decode_path() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let auto_case = bundle.case_by_id("det_auto_en_short").unwrap();
    let auto_request = auto_case.build_generation_request().unwrap();
    let mut long_request = auto_case.build_generation_request().unwrap();
    long_request.durations = vec![Some(31.0)];
    let pipeline = cuda_f32_pipeline();

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

    let actual = pipeline.generate(&request).unwrap();
    let generated = pipeline.generate_tokens(&request).unwrap();
    let expected = generated
        .iter()
        .map(|tokens| {
            pipeline.stage1().decode_final(
                tokens,
                None,
                request.generation_config.postprocess_output,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(actual.len(), 2);
    assert_eq!(expected.len(), 2);
    assert_eq!(actual[0], expected[0]);
    assert_eq!(actual[1], expected[1]);
}

#[test]
fn phase10_cuda_stage0_only_paths_do_not_eager_load_stage1() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = cuda_f32_pipeline();
    assert!(!pipeline.stage0().is_loaded());
    assert!(!pipeline.stage1().is_loaded());

    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("debug_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let _prepared = pipeline.prepare_prompt(&request).unwrap();
    assert!(!pipeline.stage0().is_loaded());
    assert!(!pipeline.stage1().is_loaded());

    let _generated = pipeline
        .generate_stage0_from_reference_case(deterministic_reference_root(), "det_auto_en_short")
        .unwrap();
    assert!(pipeline.stage0().is_loaded());
    assert!(!pipeline.stage1().is_loaded());
}

#[test]
fn phase10_cuda_auto_device_dtype_prioritize_gpu() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = auto_pipeline();
    assert!(pipeline.stage0().device().is_cuda());
    assert_eq!(pipeline.stage0().runtime_dtype(), DType::F32);

    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = pipeline.generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 480, 5.0e-4, 8.0e-4, 0.05,
    );
}
