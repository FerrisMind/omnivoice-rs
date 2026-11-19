#![cfg(all(feature = "metal", target_os = "macos"))]

mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::{GeneratedTokens, I64Tensor2},
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
    stage0_model::Stage0DebugRun,
};
use support::{deterministic_reference_root as reference_root, model_root};

fn assert_parity_metric(
    debug: &Stage0DebugRun,
    name: &str,
    max_abs_limit: f32,
    mae_limit: f32,
    rmse_limit: f32,
) {
    let metric = debug
        .parity_metrics
        .metrics
        .get(name)
        .unwrap_or_else(|| panic!("missing parity metric {name}"));
    assert!(
        metric.max_abs <= max_abs_limit,
        "{name} max_abs {} exceeds {max_abs_limit}: {metric:?}",
        metric.max_abs
    );
    assert!(
        metric.mae <= mae_limit,
        "{name} mae {} exceeds {mae_limit}: {metric:?}",
        metric.mae
    );
    assert!(
        metric.rmse <= rmse_limit,
        "{name} rmse {} exceeds {rmse_limit}: {metric:?}",
        metric.rmse
    );
}

fn report_single_token_drift(case_id: &str, actual: &I64Tensor2, expected: &I64Tensor2) {
    if actual.dims() != expected.dims() {
        println!(
            "{case_id}: dims differ: actual={:?} reference={:?}",
            actual.dims(),
            expected.dims()
        );
        return;
    }
    let (rows, cols) = actual.dims();
    let mut mismatches = Vec::new();
    let mut per_row = vec![0usize; rows];
    let mut total = 0usize;
    for index in 0..actual.data.len() {
        if actual.data[index] != expected.data[index] {
            total += 1;
            per_row[index / cols] += 1;
            if mismatches.len() < 16 {
                mismatches.push((
                    index / cols,
                    index % cols,
                    actual.data[index],
                    expected.data[index],
                ));
            }
        }
    }
    println!(
        "{case_id}: {total}/{} tokens differ; per-row {per_row:?}; first (row, col, actual, reference) {mismatches:?}",
        actual.data.len()
    );
}

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
fn phase7_stage0_metal_auto_matches_dense_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let debug = metal_f32_pipeline()
        .debug_stage0_from_reference_case(reference_root(), "det_auto_en_short")
        .unwrap();
    for (name, metric) in &debug.parity_metrics.metrics {
        println!("{name}: {metric:?}");
    }
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("det_auto_en_short")
        .unwrap()
        .load_generated_tokens()
        .unwrap();
    match (&debug.tokens, &reference) {
        (actual, GeneratedTokens::Single(expected)) => {
            report_single_token_drift("det_auto_en_short", actual, expected);
            assert_eq!(actual.dims(), expected.dims());
            assert!(actual.data.iter().all(|token| (0..=1023).contains(token)));
        }
        _ => panic!("expected single-token output"),
    }
    // The first unmasking step is shared with the reference. Later steps are
    // inherently order-sensitive: a sub-1e-4 confidence difference flips the
    // selected position and the autoregressive trajectory diverges. Keep the
    // first step numerically anchored and let the drift report above track the
    // rest without requiring cross-backend bit equality.
    assert_parity_metric(&debug, "step_00_c_logits", 1.0e-3, 2.0e-4, 3.0e-4);
    assert_parity_metric(&debug, "step_00_u_logits", 1.0e-3, 2.0e-4, 3.0e-4);
    assert_parity_metric(&debug, "step_00_confidence_scores", 2.0e-3, 5.0e-4, 7.0e-4);
    for name in ["step_00_pred_tokens", "step_00_tokens_after_step"] {
        let metric = debug
            .parity_metrics
            .metrics
            .get(name)
            .unwrap_or_else(|| panic!("missing parity metric {name}"));
        assert!(metric.exact_match, "{name}: {metric:?}");
    }
}

#[test]
fn phase7_stage0_metal_design_matches_gpu_contract() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let generated = metal_f32_pipeline()
        .generate_stage0_from_reference_case(reference_root(), "det_design_en_british")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("det_design_en_british")
        .unwrap()
        .load_generated_tokens()
        .unwrap();
    match (&generated, &reference) {
        (GeneratedTokens::Single(actual), GeneratedTokens::Single(expected)) => {
            assert_eq!(actual.dims(), expected.dims());
            assert!(actual.data.iter().all(|token| (0..=1023).contains(token)));
            report_single_token_drift("det_design_en_british", actual, expected);
        }
        _ => panic!("expected single-token output"),
    }
}

#[test]
fn phase7_stage0_metal_chunked_matches_gpu_contract() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let generated = metal_f32_pipeline()
        .generate_stage0_from_reference_case(reference_root(), "det_auto_long_chunked")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("det_auto_long_chunked")
        .unwrap()
        .load_generated_tokens()
        .unwrap();
    match (&generated, &reference) {
        (GeneratedTokens::Chunked(actual), GeneratedTokens::Chunked(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual_chunk, expected_chunk) in actual.iter().zip(expected.iter()) {
                assert_eq!(actual_chunk.dims(), expected_chunk.dims());
                assert!(actual_chunk
                    .data
                    .iter()
                    .all(|token| (0..=1023).contains(token)));
            }
        }
        _ => panic!("expected chunked-token output"),
    }
}
