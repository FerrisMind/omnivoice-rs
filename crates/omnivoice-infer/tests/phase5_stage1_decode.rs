mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle, gpu_lock::acquire_gpu_test_lock, pipeline::Phase3Pipeline,
    runtime::RuntimeOptions,
};
use support::{model_root, reference_root};

#[cfg(feature = "cuda")]
use omnivoice_infer::runtime::{DTypeSpec, DeviceSpec};

#[cfg(feature = "cuda")]
fn cuda_f32_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Cuda(0))
            .with_dtype(DTypeSpec::F32),
    )
    .unwrap()
}

#[test]
fn stage1_loader_discovers_expected_quantizer_indices() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
    assert_eq!(
        pipeline.stage1().bundle().active_quantizer_indices(),
        &[0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn debug_stage1_raw_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
    let actual = pipeline
        .decode_stage1_raw_from_reference_case(reference_root(), "debug_auto_en_short")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("debug_auto_en_short")
        .unwrap()
        .load_decoded_raw_audio()
        .unwrap();
    let metrics = actual.parity_metrics(&reference).unwrap();

    assert_eq!(metrics.sample_rate, 24_000);
    assert_eq!(metrics.frame_count, 90_240);
    assert!(metrics.mae < 1.0e-6, "{metrics:?}");
    assert!(metrics.rmse < 1.0e-6, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-6, "{metrics:?}");
}

#[test]
fn debug_stage1_final_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
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
    assert_eq!(metrics.frame_count, 77_040);
    assert!(metrics.mae < 2.0e-5, "{metrics:?}");
    assert!(metrics.rmse < 3.0e-5, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-4, "{metrics:?}");
}

#[test]
fn clone_stage1_final_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
    let actual = pipeline
        .decode_stage1_final_from_reference_case(reference_root(), "clone_user_ref")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("clone_user_ref")
        .unwrap()
        .load_final_audio()
        .unwrap();
    let metrics = actual.parity_metrics(&reference).unwrap();

    assert_eq!(metrics.sample_rate, 24_000);
    assert_eq!(metrics.frame_count, 104_160);
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[test]
fn design_stage1_final_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
    let actual = pipeline
        .decode_stage1_final_from_reference_case(reference_root(), "design_en_british")
        .unwrap();
    let reference = ReferenceArtifactBundle::from_root(reference_root())
        .unwrap()
        .case_by_id("design_en_british")
        .unwrap()
        .load_final_audio()
        .unwrap();
    let metrics = actual.parity_metrics(&reference).unwrap();

    assert_eq!(metrics.sample_rate, 24_000);
    assert_eq!(metrics.frame_count, 142_560);
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[test]
fn chunked_stage1_final_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
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
    assert_eq!(metrics.frame_count, 652_800);
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn debug_stage1_cuda_final_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = cuda_f32_pipeline();
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
    assert_eq!(metrics.frame_count, 77_040);
    assert!(metrics.mae < 2.0e-5, "{metrics:?}");
    assert!(metrics.rmse < 3.0e-5, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-4, "{metrics:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn chunked_stage1_cuda_final_matches_reference_waveform() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = cuda_f32_pipeline();
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
    assert_eq!(metrics.frame_count, 652_800);
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}
