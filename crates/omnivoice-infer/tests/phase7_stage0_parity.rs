#![cfg(feature = "cuda")]

mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::GeneratedTokens,
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{deterministic_reference_root, model_root};

fn reference_root() -> std::path::PathBuf {
    deterministic_reference_root()
}

fn cuda_f32_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F32),
    )
    .unwrap()
}

fn cuda_f16_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F16),
    )
    .unwrap()
}

#[test]
fn phase7_stage0_cuda_f32_auto_matches_dense_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    assert_dense_debug_case(&cuda_f32_pipeline(), "det_auto_en_short");
}

#[test]
fn phase7_stage0_cuda_f32_clone_matches_dense_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    assert_dense_debug_case(&cuda_f32_pipeline(), "det_clone_user_ref");
}

#[test]
fn phase7_stage0_cuda_f32_design_matches_dense_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    assert_design_gpu_contract(&cuda_f32_pipeline(), "det_design_en_british");
}

#[test]
fn phase7_stage0_cuda_f32_chunked_matches_dense_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    assert_chunked_gpu_contract(&cuda_f32_pipeline(), "det_auto_long_chunked");
}

#[test]
fn phase7_stage0_cuda_f16_auto_matches_gpu_contract() {
    let _guard = acquire_gpu_test_lock().unwrap();
    assert_generated_contract_matches_reference(&cuda_f16_pipeline(), "det_auto_en_short");
}

#[test]
fn phase7_stage0_cuda_f16_clone_matches_gpu_contract() {
    let _guard = acquire_gpu_test_lock().unwrap();
    assert_generated_contract_matches_reference(&cuda_f16_pipeline(), "det_clone_user_ref");
}

#[test]
fn phase7_stage0_cuda_f16_design_matches_gpu_contract() {
    let _guard = acquire_gpu_test_lock().unwrap();
    assert_generated_contract_matches_reference(&cuda_f16_pipeline(), "det_design_en_british");
}

#[test]
fn phase7_stage0_cuda_f16_chunked_matches_gpu_contract() {
    let _guard = acquire_gpu_test_lock().unwrap();
    assert_generated_contract_matches_reference(&cuda_f16_pipeline(), "det_auto_long_chunked");
}

#[test]
fn phase7_stage0_cuda_f32_request_auto_matches_reference_tokens() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(
        reference_root(),
    )
    .unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = cuda_f32_pipeline().generate_tokens(&request).unwrap();
    let expected = case.load_generated_tokens().unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0], expected);
}

fn assert_dense_debug_case(pipeline: &Phase3Pipeline, case_id: &str) {
    let reference_bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = reference_bundle.case_by_id(case_id).unwrap();
    let debug_config = case.load_stage0_debug_config().unwrap();
    let debug = pipeline
        .debug_stage0_from_reference_case(reference_root(), case_id)
        .unwrap();
    let generated = pipeline
        .generate_stage0_from_reference_case(reference_root(), case_id)
        .unwrap();
    let reference_generated = case.load_generated_tokens().unwrap();

    assert_eq!(generated, reference_generated, "{case_id}");
    match reference_generated {
        GeneratedTokens::Single(reference_tokens) => {
            assert_eq!(debug.tokens, reference_tokens, "{case_id}");
            assert_eq!(
                debug.debug_capture.final_tokens, reference_tokens,
                "{case_id}"
            );
        }
        GeneratedTokens::Chunked(reference_chunks) => {
            let reference_last_chunk = reference_chunks.last().unwrap();
            assert_eq!(debug.tokens, *reference_last_chunk, "{case_id}");
            assert_eq!(
                debug.debug_capture.final_tokens, *reference_last_chunk,
                "{case_id}"
            );
        }
    }

    assert_metric_tight(&debug, "inputs_embeds", 1.0e-6);
    for layer in debug_config.capture_hidden_layers {
        assert_hidden_metric_tight(&debug, &format!("hidden_layer_{layer:02}"));
    }
    if debug_config.capture_final_hidden {
        assert_hidden_metric_tight(&debug, "final_hidden");
    }
    for step in debug_config.capture_steps {
        assert_metric_tight(&debug, &format!("step_{step:02}_c_logits"), 5.0e-4);
        assert_metric_tight(&debug, &format!("step_{step:02}_u_logits"), 5.0e-4);
        assert_metric_tight(&debug, &format!("step_{step:02}_confidence_scores"), 8.0e-4);
        assert_metric_exact(
            &debug,
            &format!("step_{step:02}_batch_input_ids_before_step"),
        );
        assert_metric_exact(&debug, &format!("step_{step:02}_tokens_after_step"));
    }
    assert_metric_exact(&debug, "final_tokens");
}

fn assert_metric_tight(
    debug: &omnivoice_infer::stage0_model::Stage0DebugRun,
    name: &str,
    max_abs_limit: f32,
) {
    let metric = debug
        .parity_metrics
        .metrics
        .get(name)
        .unwrap_or_else(|| panic!("missing metric {name}"));
    assert!(metric.max_abs <= max_abs_limit, "{name}: {metric:?}");
    assert!(metric.mae <= max_abs_limit, "{name}: {metric:?}");
    assert!(metric.rmse <= max_abs_limit, "{name}: {metric:?}");
}

fn assert_metric_exact(debug: &omnivoice_infer::stage0_model::Stage0DebugRun, name: &str) {
    let metric = debug
        .parity_metrics
        .metrics
        .get(name)
        .unwrap_or_else(|| panic!("missing metric {name}"));
    assert!(metric.exact_match, "{name}: {metric:?}");
}

fn assert_hidden_metric_tight(debug: &omnivoice_infer::stage0_model::Stage0DebugRun, name: &str) {
    let metric = debug
        .parity_metrics
        .metrics
        .get(name)
        .unwrap_or_else(|| panic!("missing metric {name}"));
    assert!(metric.max_abs <= 3.2e-2, "{name}: {metric:?}");
    assert!(metric.mae <= 6.0e-4, "{name}: {metric:?}");
    assert!(metric.rmse <= 6.0e-4, "{name}: {metric:?}");
}

fn assert_generated_contract_matches_reference(pipeline: &Phase3Pipeline, case_id: &str) {
    let generated = pipeline
        .generate_stage0_from_reference_case(reference_root(), case_id)
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id(case_id)
        .unwrap()
        .load_generated_tokens()
        .unwrap();
    match (generated, reference) {
        (GeneratedTokens::Single(actual), GeneratedTokens::Single(expected)) => {
            assert_eq!(actual.dims(), expected.dims(), "{case_id}");
            assert_token_domain(&actual.data, case_id);
        }
        (GeneratedTokens::Chunked(actual), GeneratedTokens::Chunked(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{case_id}");
            for (index, (actual_chunk, expected_chunk)) in
                actual.iter().zip(expected.iter()).enumerate()
            {
                assert_eq!(
                    actual_chunk.dims(),
                    expected_chunk.dims(),
                    "{case_id} chunk {index}"
                );
                assert_token_domain(&actual_chunk.data, case_id);
            }
        }
        (actual, expected) => panic!(
            "{case_id}: generated kind {:?} does not match reference kind {:?}",
            token_kind(&actual),
            token_kind(&expected)
        ),
    }
}

fn assert_design_gpu_contract(pipeline: &Phase3Pipeline, case_id: &str) {
    let reference_bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = reference_bundle.case_by_id(case_id).unwrap();
    let debug = pipeline
        .debug_stage0_from_reference_case(reference_root(), case_id)
        .unwrap();
    let generated = pipeline
        .generate_stage0_from_reference_case(reference_root(), case_id)
        .unwrap();
    let reference = case.load_generated_tokens().unwrap();

    match (&generated, &reference) {
        (GeneratedTokens::Single(actual), GeneratedTokens::Single(expected)) => {
            assert_eq!(actual.dims(), expected.dims(), "{case_id}");
            assert_token_domain(&actual.data, case_id);
        }
        _ => panic!("{case_id}: expected single-token output"),
    }

    assert_metric_tight(&debug, "inputs_embeds", 1.0e-6);
    for layer in case
        .load_stage0_debug_config()
        .unwrap()
        .capture_hidden_layers
    {
        assert_hidden_metric_tight(&debug, &format!("hidden_layer_{layer:02}"));
    }
    assert_hidden_metric_tight(&debug, "final_hidden");
    for step in 0..30 {
        assert_metric_tight(&debug, &format!("step_{step:02}_c_logits"), 5.0e-4);
        assert_metric_tight(&debug, &format!("step_{step:02}_u_logits"), 5.0e-4);
        assert_metric_tight(&debug, &format!("step_{step:02}_confidence_scores"), 8.0e-4);
        assert_metric_exact(
            &debug,
            &format!("step_{step:02}_batch_input_ids_before_step"),
        );
        assert_metric_exact(&debug, &format!("step_{step:02}_tokens_after_step"));
    }

    assert_stage1_audio_contract(pipeline, case_id, &generated, 2.0e-3, 6.0e-3, 0.35);
}

fn assert_chunked_gpu_contract(pipeline: &Phase3Pipeline, case_id: &str) {
    let generated = pipeline
        .generate_stage0_from_reference_case(reference_root(), case_id)
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id(case_id)
        .unwrap()
        .load_generated_tokens()
        .unwrap();

    match (&generated, &reference) {
        (GeneratedTokens::Chunked(actual), GeneratedTokens::Chunked(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{case_id}");
            for (index, (actual_chunk, expected_chunk)) in
                actual.iter().zip(expected.iter()).enumerate()
            {
                assert_eq!(
                    actual_chunk.dims(),
                    expected_chunk.dims(),
                    "{case_id} chunk {index}"
                );
                assert_token_domain(&actual_chunk.data, case_id);
            }
        }
        _ => panic!("{case_id}: expected chunked-token output"),
    }
}

fn assert_stage1_audio_contract(
    pipeline: &Phase3Pipeline,
    case_id: &str,
    generated: &GeneratedTokens,
    mae_limit: f32,
    rmse_limit: f32,
    max_abs_limit: f32,
) {
    let reference_bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = reference_bundle.case_by_id(case_id).unwrap();
    let metadata = case.load_prepared_metadata().unwrap();
    let reference_audio = case.load_final_audio().unwrap();
    let actual_audio = pipeline
        .stage1()
        .decode_final(generated, metadata.ref_rms, metadata.postprocess_output)
        .unwrap();
    let metrics = actual_audio.parity_metrics(&reference_audio).unwrap();

    assert_eq!(metrics.sample_rate, 24_000, "{case_id}");
    assert_eq!(
        metrics.frame_count,
        reference_audio.frame_count(),
        "{case_id}"
    );
    assert!(metrics.mae <= mae_limit, "{case_id}: {metrics:?}");
    assert!(metrics.rmse <= rmse_limit, "{case_id}: {metrics:?}");
    assert!(metrics.max_abs <= max_abs_limit, "{case_id}: {metrics:?}");
}

fn token_kind(tokens: &GeneratedTokens) -> &'static str {
    match tokens {
        GeneratedTokens::Single(_) => "single",
        GeneratedTokens::Chunked(_) => "chunked",
    }
}

fn assert_token_domain(tokens: &[i64], case_id: &str) {
    for (index, token) in tokens.iter().enumerate() {
        assert!(
            (0..=1023).contains(token),
            "{case_id}: token {token} at index {index} is out of domain"
        );
    }
}
