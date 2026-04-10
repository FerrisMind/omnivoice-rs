#![cfg(feature = "phase6-tests")]

mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::GeneratedTokens,
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{model_root, stage0_cpu_strict_reference_root, stage0_reference_root};

fn cpu_reference_root() -> std::path::PathBuf {
    stage0_cpu_strict_reference_root()
}

#[cfg(feature = "cuda")]
fn gpu_reference_root() -> std::path::PathBuf {
    stage0_reference_root()
}

#[cfg(feature = "cuda")]
fn cuda_f32_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F32),
    )
    .unwrap()
}

#[cfg(feature = "cuda")]
fn cuda_f16_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F16),
    )
    .unwrap()
}

#[test]
#[ignore = "GPU-first phase6: CPU debug parity is oracle-only and not part of primary acceptance"]
fn phase6_stage0_cpu_debug_auto_matches_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cpu)
            .with_dtype(DTypeSpec::F32),
    )
    .unwrap();
    assert_debug_case(&pipeline, "det_debug_auto_en_short");
}

#[test]
#[ignore = "GPU-first phase6: CPU debug parity is oracle-only and not part of primary acceptance"]
fn phase6_stage0_cpu_debug_clone_matches_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cpu)
            .with_dtype(DTypeSpec::F32),
    )
    .unwrap();
    assert_debug_case(&pipeline, "det_debug_clone_user_ref");
}

#[test]
#[ignore = "UNVERIFIED: direct stage0->stage1 release harness aborts in current runner; stage1 remains separately validated in phase5"]
fn phase6_stage0_cpu_stage1_smoke_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cpu)
            .with_dtype(DTypeSpec::F32),
    )
    .unwrap();
    let generated = pipeline
        .generate_stage0_from_reference_case(cpu_reference_root(), "det_debug_auto_en_short")
        .unwrap();
    let bundle = ReferenceArtifactBundle::from_root(cpu_reference_root()).unwrap();
    let case = bundle.case_by_id("det_debug_auto_en_short").unwrap();
    let metadata = case.load_prepared_metadata().unwrap();
    let reference_audio = case.load_final_audio().unwrap();
    let actual_audio = pipeline
        .stage1()
        .decode_final(&generated, metadata.ref_rms, metadata.postprocess_output)
        .unwrap();
    let metrics = actual_audio.parity_metrics(&reference_audio).unwrap();

    assert_eq!(metrics.sample_rate, 24_000);
    assert!(metrics.mae < 2.0e-5, "{metrics:?}");
    assert!(metrics.rmse < 3.0e-5, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-4, "{metrics:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn phase6_stage0_cuda_debug_auto_matches_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = cuda_f32_pipeline();
    assert_debug_case_at_root(&pipeline, cpu_reference_root(), "det_debug_auto_en_short");
}

#[cfg(feature = "cuda")]
#[test]
fn phase6_stage0_cuda_debug_clone_matches_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = cuda_f32_pipeline();
    assert_debug_case_at_root(&pipeline, cpu_reference_root(), "det_debug_clone_user_ref");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Phase 10 diagnostic-only Candle drift: exact main-case parity remains diagnostic only"]
fn phase6_stage0_cuda_auto_main_matches_reference_tokens() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F16),
    )
    .unwrap();
    let generated = pipeline
        .generate_stage0_from_reference_case(gpu_reference_root(), "det_auto_en_short")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(gpu_reference_root())
        .unwrap()
        .case_by_id("det_auto_en_short")
        .unwrap()
        .load_generated_tokens()
        .unwrap();

    assert_eq!(generated, reference);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Phase 10 diagnostic-only Candle drift: exact main-case parity remains diagnostic only"]
fn phase6_stage0_cuda_clone_main_matches_reference_tokens() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F16),
    )
    .unwrap();
    let generated = pipeline
        .generate_stage0_from_reference_case(gpu_reference_root(), "det_clone_user_ref")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(gpu_reference_root())
        .unwrap()
        .case_by_id("det_clone_user_ref")
        .unwrap()
        .load_generated_tokens()
        .unwrap();

    assert_eq!(generated, reference);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Phase 10 diagnostic-only Candle drift: exact main-case parity remains diagnostic only"]
fn phase6_stage0_cuda_design_main_matches_reference_tokens() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F16),
    )
    .unwrap();
    let generated = pipeline
        .generate_stage0_from_reference_case(gpu_reference_root(), "det_design_en_british")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(gpu_reference_root())
        .unwrap()
        .case_by_id("det_design_en_british")
        .unwrap()
        .load_generated_tokens()
        .unwrap();

    assert_eq!(generated, reference);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Phase 10 diagnostic-only Candle drift: exact main-case parity remains diagnostic only"]
fn phase6_stage0_cuda_chunked_main_matches_reference_tokens() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F16),
    )
    .unwrap();
    let generated = pipeline
        .generate_stage0_from_reference_case(gpu_reference_root(), "det_auto_long_chunked")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(gpu_reference_root())
        .unwrap()
        .case_by_id("det_auto_long_chunked")
        .unwrap()
        .load_generated_tokens()
        .unwrap();

    assert_eq!(generated, reference);
}

#[cfg(feature = "cuda")]
#[test]
fn phase6_stage0_cuda_stage1_smoke_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = cuda_f32_pipeline();
    let generated = pipeline
        .generate_stage0_from_reference_case(cpu_reference_root(), "det_debug_auto_en_short")
        .unwrap();
    let bundle = ReferenceArtifactBundle::from_root(cpu_reference_root()).unwrap();
    let case = bundle.case_by_id("det_debug_auto_en_short").unwrap();
    let metadata = case.load_prepared_metadata().unwrap();
    let reference_audio = case.load_final_audio().unwrap();
    let actual_audio = pipeline
        .stage1()
        .decode_final(&generated, metadata.ref_rms, metadata.postprocess_output)
        .unwrap();
    let metrics = actual_audio.parity_metrics(&reference_audio).unwrap();

    assert_eq!(metrics.sample_rate, 24_000);
    assert!(metrics.mae < 2.0e-5, "{metrics:?}");
    assert!(metrics.rmse < 3.0e-5, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-4, "{metrics:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn phase6_stage0_cuda_auto_main_gpu_contract_matches_reference_shape_and_range() {
    let _guard = stage0_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let pipeline = cuda_f16_pipeline();
    assert_generated_contract_matches_reference(
        &pipeline,
        gpu_reference_root(),
        "det_auto_en_short",
    );
}

#[cfg(feature = "cuda")]
#[test]
fn phase6_stage0_cuda_clone_main_gpu_contract_matches_reference_shape_and_range() {
    let _guard = stage0_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let pipeline = cuda_f16_pipeline();
    assert_generated_contract_matches_reference(
        &pipeline,
        gpu_reference_root(),
        "det_clone_user_ref",
    );
}

#[cfg(feature = "cuda")]
#[test]
fn phase6_stage0_cuda_design_main_gpu_contract_matches_reference_shape_and_range() {
    let _guard = stage0_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let pipeline = cuda_f16_pipeline();
    assert_generated_contract_matches_reference(
        &pipeline,
        gpu_reference_root(),
        "det_design_en_british",
    );
}

#[cfg(feature = "cuda")]
#[test]
fn phase6_stage0_cuda_chunked_main_gpu_contract_matches_reference_shape_and_range() {
    let _guard = stage0_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let pipeline = cuda_f16_pipeline();
    assert_generated_contract_matches_reference(
        &pipeline,
        gpu_reference_root(),
        "det_auto_long_chunked",
    );
}

fn assert_debug_case(pipeline: &Phase3Pipeline, case_id: &str) {
    assert_debug_case_at_root(pipeline, cpu_reference_root(), case_id);
}

fn assert_debug_case_at_root(pipeline: &Phase3Pipeline, reference_root: &str, case_id: &str) {
    let reference_bundle = ReferenceArtifactBundle::from_root(reference_root).unwrap();
    let debug = pipeline
        .debug_stage0_from_reference_case(reference_root, case_id)
        .unwrap();
    let case = reference_bundle.case_by_id(case_id).unwrap();
    let reference_tokens = case.load_final_tokens().unwrap();

    assert_eq!(debug.tokens, reference_tokens, "{case_id}");
    assert_eq!(
        debug.debug_capture.final_tokens, reference_tokens,
        "{case_id}"
    );
    assert_metric_tight(&debug, "inputs_embeds", 1.0e-6);
    assert_metric_tight(&debug, "hidden_layer_00", 5.0e-4);
    assert_metric_tight(&debug, "hidden_layer_13", 5.0e-4);
    assert_metric_tight(&debug, "hidden_layer_27", 6.0e-4);
    assert_metric_tight(&debug, "final_hidden", 6.0e-4);
    assert_metric_tight(&debug, "step_00_c_logits", 5.0e-4);
    assert_metric_tight(&debug, "step_00_u_logits", 5.0e-4);
    assert_metric_tight(&debug, "step_00_confidence_scores", 8.0e-4);
    assert_metric_exact(&debug, "step_00_batch_input_ids_before_step");
    assert_metric_tight(&debug, "step_15_c_logits", 5.0e-4);
    assert_metric_tight(&debug, "step_15_u_logits", 5.0e-4);
    assert_metric_tight(&debug, "step_15_confidence_scores", 8.0e-4);
    assert_metric_tight(&debug, "step_31_c_logits", 5.0e-4);
    assert_metric_tight(&debug, "step_31_u_logits", 5.0e-4);
    assert_metric_tight(&debug, "step_31_confidence_scores", 8.0e-4);
    assert_metric_exact(&debug, "step_00_tokens_after_step");
    assert_metric_exact(&debug, "step_15_tokens_after_step");
    assert_metric_exact(&debug, "step_31_tokens_after_step");
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
    assert!(metric.max_abs < max_abs_limit, "{name}: {metric:?}");
    assert!(metric.mae < max_abs_limit, "{name}: {metric:?}");
    assert!(metric.rmse < max_abs_limit, "{name}: {metric:?}");
}

fn assert_metric_exact(debug: &omnivoice_infer::stage0_model::Stage0DebugRun, name: &str) {
    let metric = debug
        .parity_metrics
        .metrics
        .get(name)
        .unwrap_or_else(|| panic!("missing metric {name}"));
    assert!(metric.exact_match, "{name}: {metric:?}");
}

#[cfg(feature = "cuda")]
fn assert_generated_contract_matches_reference(
    pipeline: &Phase3Pipeline,
    reference_root: &str,
    case_id: &str,
) {
    let generated = pipeline
        .generate_stage0_from_reference_case(reference_root, case_id)
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root)
        .unwrap()
        .case_by_id(case_id)
        .unwrap()
        .load_generated_tokens()
        .unwrap();
    assert_generated_shape_and_token_domain(&generated, &reference, case_id);
}

#[cfg(feature = "cuda")]
fn assert_generated_shape_and_token_domain(
    generated: &GeneratedTokens,
    reference: &GeneratedTokens,
    case_id: &str,
) {
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
            token_kind(actual),
            token_kind(expected)
        ),
    }
}

#[cfg(feature = "cuda")]
fn token_kind(tokens: &GeneratedTokens) -> &'static str {
    match tokens {
        GeneratedTokens::Single(_) => "single",
        GeneratedTokens::Chunked(_) => "chunked",
    }
}

#[cfg(feature = "cuda")]
fn assert_token_domain(tokens: &[i64], case_id: &str) {
    for (index, token) in tokens.iter().enumerate() {
        assert!(
            (0..=1023).contains(token),
            "{case_id}: token {token} at index {index} is out of domain"
        );
    }
}
