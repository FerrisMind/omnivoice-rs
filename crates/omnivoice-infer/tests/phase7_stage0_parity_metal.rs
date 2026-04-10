#![cfg(all(feature = "metal", target_os = "macos"))]

mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::GeneratedTokens,
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{deterministic_reference_root as reference_root, model_root};

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
    let generated = metal_f32_pipeline()
        .generate_stage0_from_reference_case(reference_root(), "det_auto_en_short")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("det_auto_en_short")
        .unwrap()
        .load_generated_tokens()
        .unwrap();
    assert_eq!(generated, reference);
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
