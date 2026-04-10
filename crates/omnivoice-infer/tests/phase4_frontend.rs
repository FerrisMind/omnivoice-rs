mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    audio_input::ReferenceAudioProcessor,
    contracts::{GenerationRequest, PreparedPromptSequence},
    frontend::{
        add_punctuation, chunk_text_punctuation, combine_text, resolve_instruct, resolve_language,
        Frontend, RuleDurationEstimator,
    },
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{model_root, reference_root, ref_audio_path, stage0_reference_root};

#[test]
fn language_resolution_matches_omnivoice_lang_map() {
    assert_eq!(resolve_language(Some("English")).as_deref(), Some("en"));
    assert_eq!(resolve_language(Some("Chinese")).as_deref(), Some("zh"));
    assert_eq!(resolve_language(Some("Cantonese")).as_deref(), Some("yue"));
    assert_eq!(resolve_language(Some("en")).as_deref(), Some("en"));
    assert_eq!(resolve_language(Some("None")), None);
    assert_eq!(resolve_language(Some("not-a-language")), None);
}

#[test]
fn instruct_resolution_matches_omnivoice_rules() {
    assert_eq!(
        resolve_instruct(Some("female, low pitch, british accent"), false).unwrap(),
        Some("female, low pitch, british accent".to_string())
    );
    assert_eq!(
        resolve_instruct(Some("female, low pitch"), true).unwrap(),
        Some("女，低音调".to_string())
    );
    assert_eq!(
        resolve_instruct(Some("男，河南话"), false).unwrap(),
        Some("男，河南话".to_string())
    );

    let conflict = resolve_instruct(Some("british accent, 河南话"), false).unwrap_err();
    assert!(conflict
        .to_string()
        .contains("Cannot mix Chinese dialect and English accent"));

    let suggestion = resolve_instruct(Some("british accnt"), false).unwrap_err();
    assert!(suggestion
        .to_string()
        .contains("did you mean 'british accent'"));
}

#[test]
fn text_preprocessing_matches_omnivoice_helpers() {
    assert_eq!(add_punctuation("Hello there"), "Hello there.");
    assert_eq!(add_punctuation("你好"), "你好。");
    assert_eq!(add_punctuation("Hello there!"), "Hello there!");

    assert_eq!(
        combine_text("你好 [question-en]\nworld", Some(" reference ")),
        "reference你好[question-en].world"
    );
    assert_eq!(combine_text("hello [laughter]", None), "hello [laughter]");

    assert_eq!(
        chunk_text_punctuation("Mr. Smith arrived. Dr. Brown stayed.", 20, Some(3)),
        vec![
            "Mr. Smith arrived.".to_string(),
            "Dr. Brown stayed.".to_string()
        ]
    );
}

#[test]
fn duration_estimator_matches_reference_values() {
    let estimator = RuleDurationEstimator;

    assert_close(
        estimator.calculate_total_weight("Hello, world."),
        11.2,
        1e-6,
    );
    assert_close(
        estimator.estimate_duration("Hello, world.", "Hello, world.", 25.0, Some(50.0), 3.0),
        39.685_026,
        1e-5,
    );
    assert_close(
        estimator.estimate_duration("你好，世界！", "Hello, world.", 25.0, Some(50.0), 3.0),
        41.706_31,
        1e-5,
    );
    assert_close(
        estimator.estimate_duration("मेरा नाम ओम्नीवॉइस है", "Hello, world.", 25.0, Some(50.0), 3.0),
        48.465_28,
        1e-5,
    );
}

#[test]
fn artifact_loader_supports_single_and_chunked_prepared_json() {
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();

    match bundle
        .case_by_id("debug_auto_en_short")
        .unwrap()
        .load_prepared_prompts()
        .unwrap()
    {
        PreparedPromptSequence::Single(prompt) => {
            assert_eq!(prompt.prompt.input_ids_dims(), (1, 8, 114));
            assert_eq!(prompt.target_start_idx, 20);
        }
        PreparedPromptSequence::Chunked(_) => panic!("expected single prepared prompt"),
    }

    match bundle
        .case_by_id("auto_long_chunked")
        .unwrap()
        .load_prepared_prompts()
        .unwrap()
    {
        PreparedPromptSequence::Single(_) => panic!("expected chunked prepared prompts"),
        PreparedPromptSequence::Chunked(chunked) => {
            assert_eq!(chunked.chunk_target_lens, vec![370, 177, 123]);
            assert_eq!(chunked.prompts.len(), 3);
            assert_eq!(
                chunked.prompts[0].mode,
                omnivoice_infer::contracts::GenerationMode::Auto
            );
            assert_eq!(
                chunked.prompts[1].mode,
                omnivoice_infer::contracts::GenerationMode::Clone
            );
            assert_eq!(
                chunked.prompts[2].mode,
                omnivoice_infer::contracts::GenerationMode::Clone
            );
            assert_eq!(chunked.prompts[0].target_start_idx, 83);
            assert_eq!(chunked.prompts[1].target_start_idx, 516);
            assert_eq!(chunked.prompts[2].target_start_idx, 494);
            assert_eq!(chunked.prompts[0].prompt.input_ids_dims(), (1, 8, 453));
            assert_eq!(chunked.prompts[1].prompt.input_ids_dims(), (1, 8, 693));
            assert_eq!(chunked.prompts[2].prompt.input_ids_dims(), (1, 8, 617));
        }
    }
}

#[test]
fn frontend_chunk_plan_matches_reference_auto_long_case() {
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("auto_long_chunked").unwrap();
    let definition = case.load_case_definition().unwrap();
    let expected = match case.load_prepared_prompts().unwrap() {
        PreparedPromptSequence::Chunked(chunked) => chunked,
        PreparedPromptSequence::Single(_) => panic!("expected chunked prepared prompts"),
    };

    let frontend = Frontend::from_model_root(model_root()).unwrap();
    let task = frontend.build_task(&definition.request).unwrap();
    let planned_chunks = frontend.chunk_text(
        &task.texts[0],
        task.target_lens[0],
        definition.audio_chunk_duration,
    );

    assert_eq!(frontend.frame_rate(), 25);
    assert_eq!(task.target_lens, vec![875]);
    assert_eq!(planned_chunks, expected.chunk_texts);
}

#[test]
fn phase10_duration_override_recomputes_effective_speed() {
    let frontend = Frontend::from_model_root(model_root()).unwrap();
    let text = "Hello, world.";
    let estimated_unit_speed = frontend.estimate_target_tokens(text, None, None, 1.0);
    let expected_duration_target = frontend.frame_rate() * 2;
    let expected_effective_speed = estimated_unit_speed as f32 / expected_duration_target as f32;

    let mut speed_only = GenerationRequest::new_text_only(text);
    speed_only.speeds = vec![Some(2.0)];
    let speed_only_task = frontend.build_task(&speed_only).unwrap();
    assert_eq!(
        speed_only_task.target_lens,
        vec![frontend.estimate_target_tokens(text, None, None, 2.0)]
    );
    assert_eq!(speed_only_task.speed, vec![2.0]);

    let mut duration_only = GenerationRequest::new_text_only(text);
    duration_only.durations = vec![Some(2.0)];
    let duration_only_task = frontend.build_task(&duration_only).unwrap();
    assert_eq!(
        duration_only_task.target_lens,
        vec![expected_duration_target]
    );
    assert_close(
        duration_only_task.speed[0],
        expected_effective_speed,
        1.0e-6,
    );

    let mut duration_and_speed = GenerationRequest::new_text_only(text);
    duration_and_speed.durations = vec![Some(2.0)];
    duration_and_speed.speeds = vec![Some(2.0)];
    let duration_and_speed_task = frontend.build_task(&duration_and_speed).unwrap();
    assert_eq!(
        duration_and_speed_task.target_lens,
        vec![expected_duration_target]
    );
    assert_close(
        duration_and_speed_task.speed[0],
        expected_effective_speed,
        1.0e-6,
    );
}

#[test]
fn reference_audio_preprocessing_matches_python_prompt_contract() {
    let processor = ReferenceAudioProcessor::new(24_000, 960);
    let prepared = processor
        .prepare_prompt_audio(
            &omnivoice_infer::contracts::ReferenceAudioInput::from_path(
                ref_audio_path().display().to_string(),
            ),
            Some("State-of-the-art text-to-speech model for 600+ languages, supporting"),
            true,
        )
        .unwrap();

    assert_eq!(prepared.waveform.len(), 101_760);
    assert_close(prepared.ref_rms.unwrap(), 0.059_361_65, 5.0e-6);
    assert_eq!(&prepared.waveform[..10], &[0.0; 10]);
}

#[test]
fn prepare_prompt_matches_deterministic_reference_batch() {
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cpu)
            .with_dtype(DTypeSpec::F32),
    )
    .unwrap();
    let actual = pipeline.prepare_prompt(&request).unwrap();
    let expected_prompt = match case.load_prepared_prompts().unwrap() {
        PreparedPromptSequence::Single(prompt) => prompt,
        PreparedPromptSequence::Chunked(_) => panic!("expected single prepared prompt"),
    };
    let expected_batched = omnivoice_infer::stage0_loop::pack_cfg_batch(
        std::slice::from_ref(&expected_prompt),
        &[expected_prompt.target_length],
    )
    .unwrap();
    let expected = pipeline
        .stage0()
        .prepare_batch(
            &expected_batched,
            &[expected_prompt.total_length],
            &[expected_prompt.target_length],
        )
        .unwrap();

    assert_eq!(actual.target_lens, expected.target_lens);
    assert_eq!(actual.cond_lens, expected.cond_lens);
    assert_eq!(actual.runtime_dtype, expected.runtime_dtype);
    assert_eq!(
        actual
            .input_ids
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap(),
        expected
            .input_ids
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap(),
    );
    assert_eq!(
        actual
            .audio_mask
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        expected
            .audio_mask
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
    );
    assert_eq!(
        actual
            .attention_mask
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        expected
            .attention_mask
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
    );
    assert_eq!(
        actual
            .tokens_init
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap(),
        expected
            .tokens_init
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap(),
    );
}

fn deterministic_reference_root() -> std::path::PathBuf {
    stage0_reference_root()
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "expected {expected}, got {actual}, delta {delta} > {tolerance}"
    );
}
