#![cfg(feature = "cuda")]

mod support;

use omnivoice_infer::{
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

#[test]
fn phase7_stage1_cuda_dense_auto_matches_reference() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let debug = cuda_f32_pipeline()
        .debug_stage1_from_reference_case(reference_root(), "det_auto_en_short")
        .unwrap();
    assert!(debug.tensor_metrics.contains_key("quantizer_output"));
    assert!(debug.tensor_metrics.contains_key("project_out"));
    assert!(debug.tensor_metrics.contains_key("fc2_output"));
    assert!(debug.tensor_metrics.contains_key("decoder_input"));
    assert!(debug.tensor_metrics.contains_key("decoder_block_00"));
    assert_eq!(debug.raw_audio_metrics.sample_rate, 24_000);
    assert_eq!(debug.final_audio_metrics.sample_rate, 24_000);
}
