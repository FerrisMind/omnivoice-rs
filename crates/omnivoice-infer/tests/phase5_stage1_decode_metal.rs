#![cfg(all(feature = "metal", target_os = "macos"))]

mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{model_root, reference_root};

fn metal_f32_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Metal)
            .with_dtype(DTypeSpec::F32),
    )
    .unwrap()
}

#[test]
fn debug_stage1_metal_final_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = metal_f32_pipeline();
    let actual = pipeline
        .decode_stage1_final_from_reference_case(reference_root(), "debug_auto_en_short")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("debug_auto_en_short")
        .unwrap()
        .load_final_audio()
        .unwrap();
    let metrics = actual.parity_metrics(&reference).unwrap();
    assert_eq!(metrics.sample_rate, 24_000);
    assert!(metrics.mae < 2.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 3.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-3, "{metrics:?}");
}

#[test]
fn chunked_stage1_metal_final_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = metal_f32_pipeline();
    let actual = pipeline
        .decode_stage1_final_from_reference_case(reference_root(), "auto_long_chunked")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("auto_long_chunked")
        .unwrap()
        .load_final_audio()
        .unwrap();
    let metrics = actual.parity_metrics(&reference).unwrap();
    assert_eq!(metrics.sample_rate, 24_000);
    assert!(metrics.mae < 2.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 3.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-3, "{metrics:?}");
}
