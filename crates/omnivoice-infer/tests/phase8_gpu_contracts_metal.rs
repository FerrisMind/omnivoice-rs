#![cfg(all(feature = "metal", target_os = "macos"))]

mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::{GeneratedTokens, GenerationRequest, ReferenceAudioInput, VoiceClonePrompt},
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{deterministic_reference_root, model_root, ref_audio_path, reference_root};

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
fn phase8_metal_chunked_inference_rejects_mixed_reference_batch() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_long_chunked").unwrap();
    let base_request = case.build_generation_request().unwrap();
    let prompt = metal_f32_pipeline()
        .create_voice_clone_prompt_from_audio(
            &ReferenceAudioInput::from_path(ref_audio_path().display().to_string()),
            Some("State-of-the-art text-to-speech model for 600+ languages, supporting"),
            true,
            None,
        )
        .unwrap();

    let mut request = GenerationRequest::new_text_only(base_request.texts[0].clone());
    request.texts = vec![base_request.texts[0].clone(), base_request.texts[0].clone()];
    request.languages = vec![
        base_request.languages[0].clone(),
        base_request.languages[0].clone(),
    ];
    request.instructs = vec![
        base_request.instructs[0].clone(),
        base_request.instructs[0].clone(),
    ];
    request.voice_clone_prompts = vec![Some(prompt), None];
    request.ref_texts = vec![None, None];
    request.ref_audios = vec![None, None];
    request.speeds = vec![base_request.speeds[0], base_request.speeds[0]];
    request.durations = vec![base_request.durations[0], base_request.durations[0]];
    request.generation_config = base_request.generation_config.clone();

    let error = metal_f32_pipeline().generate_tokens(&request).unwrap_err();
    assert!(error.to_string().contains(
        "chunked inference requires all items to either have or not have reference audio"
    ));
}

#[test]
fn phase8_metal_long_form_auto_reuses_first_chunk_reference_for_followup_chunks() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = metal_f32_pipeline();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_long_chunked").unwrap();
    let request = case.build_generation_request().unwrap();
    let full = pipeline.generate_tokens(&request).unwrap();
    let chunks = expect_chunked(full.into_iter().next().unwrap());
    let task = pipeline.frontend().build_task(&request).unwrap();
    let chunk_texts = pipeline.frontend().chunk_text(
        &task.texts[0],
        task.target_lens[0],
        request.generation_config.audio_chunk_duration,
    );

    assert_eq!(chunk_texts.len(), chunks.len());

    let first_chunk = expect_single(generate_chunk_request(
        &pipeline,
        &chunk_texts[0],
        task.langs[0].clone(),
        task.instructs[0].clone(),
        None,
        None,
        task.speed[0],
        request.generation_config.clone(),
    ));
    assert_eq!(first_chunk, chunks[0]);

    for (index, expected_chunk) in chunks.iter().enumerate().skip(1) {
        let actual = expect_single(generate_chunk_request(
            &pipeline,
            &chunk_texts[index],
            task.langs[0].clone(),
            task.instructs[0].clone(),
            Some(chunks[0].clone()),
            Some(chunk_texts[0].clone()),
            task.speed[0],
            request.generation_config.clone(),
        ));
        assert_eq!(&actual, expected_chunk, "chunk {index}");
    }
}

#[test]
fn phase8_metal_long_form_clone_preserves_original_reference_for_each_chunk() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = metal_f32_pipeline();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_long_chunked").unwrap();
    let base_request = case.build_generation_request().unwrap();
    let prompt = pipeline
        .create_voice_clone_prompt_from_audio(
            &ReferenceAudioInput::from_path(ref_audio_path().display().to_string()),
            Some("State-of-the-art text-to-speech model for 600+ languages, supporting"),
            true,
            None,
        )
        .unwrap();
    let mut request = base_request.clone();
    request.voice_clone_prompts = vec![Some(prompt.clone())];
    request.ref_audios = vec![None];
    request.ref_texts = vec![None];

    let full = pipeline.generate_tokens(&request).unwrap();
    let chunks = expect_chunked(full.into_iter().next().unwrap());
    let task = pipeline.frontend().build_task(&request).unwrap();
    let chunk_texts = pipeline.frontend().chunk_text(
        &task.texts[0],
        task.target_lens[0],
        request.generation_config.audio_chunk_duration,
    );

    assert_eq!(chunk_texts.len(), chunks.len());

    for (index, expected_chunk) in chunks.iter().enumerate() {
        let actual = expect_single(generate_chunk_request(
            &pipeline,
            &chunk_texts[index],
            task.langs[0].clone(),
            task.instructs[0].clone(),
            Some(prompt.ref_audio_tokens.clone()),
            Some(prompt.ref_text.clone()),
            task.speed[0],
            request.generation_config.clone(),
        ));
        assert_eq!(&actual, expected_chunk, "chunk {index}");
    }
}

#[test]
fn phase8_metal_raw_reference_and_prebuilt_prompt_generate_identical_tokens() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = metal_f32_pipeline();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let mut raw_request = case.build_generation_request().unwrap();
    raw_request.ref_audios = vec![Some(ReferenceAudioInput::from_path(
        ref_audio_path().display().to_string(),
    ))];
    raw_request.voice_clone_prompts = vec![None];

    let prompt = pipeline
        .create_voice_clone_prompt_from_audio(
            &ReferenceAudioInput::from_path(ref_audio_path().display().to_string()),
            raw_request.ref_texts[0].as_deref(),
            raw_request.generation_config.preprocess_prompt,
            None,
        )
        .unwrap();

    let mut prompt_request = raw_request.clone();
    prompt_request.ref_audios = vec![None];
    prompt_request.ref_texts = vec![None];
    prompt_request.voice_clone_prompts = vec![Some(prompt)];

    assert_eq!(
        pipeline.generate_tokens(&raw_request).unwrap(),
        pipeline.generate_tokens(&prompt_request).unwrap()
    );
}

#[allow(clippy::too_many_arguments)]
fn generate_chunk_request(
    pipeline: &Phase3Pipeline,
    text: &str,
    language: Option<String>,
    instruct: Option<String>,
    ref_audio_tokens: Option<omnivoice_infer::contracts::I64Tensor2>,
    ref_text: Option<String>,
    speed: f32,
    generation_config: omnivoice_infer::contracts::GenerationConfig,
) -> GeneratedTokens {
    let mut request = GenerationRequest::new_text_only(text.to_string());
    request.languages = vec![language];
    request.instructs = vec![instruct];
    request.ref_texts = vec![None];
    request.ref_audios = vec![None];
    request.voice_clone_prompts = vec![ref_audio_tokens.map(|tokens| VoiceClonePrompt {
        ref_audio_tokens: tokens,
        ref_text: ref_text.unwrap_or_default(),
        ref_rms: None,
    })];
    request.speeds = vec![Some(speed)];
    request.generation_config = generation_config;
    pipeline
        .generate_tokens(&request)
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
}

fn expect_single(tokens: GeneratedTokens) -> omnivoice_infer::contracts::I64Tensor2 {
    match tokens {
        GeneratedTokens::Single(tokens) => tokens,
        GeneratedTokens::Chunked(_) => panic!("expected single token tensor"),
    }
}

fn expect_chunked(tokens: GeneratedTokens) -> Vec<omnivoice_infer::contracts::I64Tensor2> {
    match tokens {
        GeneratedTokens::Single(_) => panic!("expected chunked token tensors"),
        GeneratedTokens::Chunked(chunks) => chunks,
    }
}
