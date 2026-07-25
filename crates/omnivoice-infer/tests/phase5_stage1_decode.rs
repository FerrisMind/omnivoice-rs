mod support;

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle, contracts::GeneratedTokens,
    gpu_lock::acquire_gpu_test_lock, pipeline::Phase3Pipeline, runtime::RuntimeOptions,
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
fn stage1_loader_discovers_expected_quantizer_indices() {
    if !support::model_fixture_available() {
        return;
    }
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
    assert_eq!(
        pipeline.stage1().bundle().active_quantizer_indices(),
        &[0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn debug_stage1_raw_matches_reference_waveform() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("debug_auto_en_short")
    {
        return;
    }
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
    assert_eq!(metrics.frame_count, reference.frame_count());
    // GPU-only v2 fixtures are frozen on CUDA; default stage1 path may run on
    // another device, so allow Candle/Torch DAC numerical drift rather than
    // bit-exact host decode limits from older CPU-strict oracles.
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[test]
fn debug_stage1_final_matches_reference_waveform() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("debug_auto_en_short")
    {
        return;
    }
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
    assert_eq!(metrics.frame_count, reference.frame_count());
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[test]
fn clone_stage1_final_matches_reference_waveform() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("clone_user_ref")
    {
        return;
    }
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
    assert_eq!(metrics.frame_count, reference.frame_count());
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[test]
fn design_stage1_final_matches_reference_waveform() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("design_en_british")
    {
        return;
    }
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
    assert_eq!(metrics.frame_count, reference.frame_count());
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[test]
fn chunked_stage1_final_matches_reference_waveform() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("auto_long_chunked")
    {
        return;
    }
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
    assert_eq!(metrics.frame_count, reference.frame_count());
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[test]
fn design_stage1_tensor_decode_matches_generated_tokens_decode() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("design_en_british")
    {
        return;
    }
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("design_en_british").unwrap();
    let metadata = case.load_prepared_metadata().unwrap();
    let GeneratedTokens::Single(tokens) = case.load_generated_tokens().unwrap() else {
        panic!("expected single token tensor");
    };

    let prepared = pipeline
        .stage1()
        .prepare_decode(&tokens, metadata.ref_rms)
        .unwrap();
    let actual = pipeline
        .stage1()
        .decode_final_tensor(
            &prepared.tokens,
            metadata.ref_rms,
            metadata.postprocess_output,
        )
        .unwrap();
    let expected = pipeline
        .stage1()
        .decode_final(
            &GeneratedTokens::Single(tokens),
            metadata.ref_rms,
            metadata.postprocess_output,
        )
        .unwrap();
    let metrics = actual.parity_metrics(&expected).unwrap();

    assert_eq!(metrics.sample_rate, 24_000);
    assert_eq!(metrics.frame_count, expected.frame_count());
    assert!(metrics.mae < 1.0e-6, "{metrics:?}");
    assert!(metrics.rmse < 1.0e-6, "{metrics:?}");
    assert!(metrics.max_abs < 1.0e-5, "{metrics:?}");
}

#[test]
fn chunked_stage1_tensor_decode_matches_generated_tokens_decode() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("auto_long_chunked")
    {
        return;
    }
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("auto_long_chunked").unwrap();
    let metadata = case.load_prepared_metadata().unwrap();
    let GeneratedTokens::Chunked(chunks) = case.load_generated_tokens().unwrap() else {
        panic!("expected chunked token tensors");
    };

    let prepared_chunks = chunks
        .iter()
        .map(|chunk| pipeline.stage1().prepare_decode(chunk, metadata.ref_rms))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let actual = pipeline
        .stage1()
        .decode_final_tensor_chunks(
            &prepared_chunks
                .iter()
                .map(|item| item.tokens.clone())
                .collect::<Vec<_>>(),
            metadata.ref_rms,
            metadata.postprocess_output,
        )
        .unwrap();
    let expected = pipeline
        .stage1()
        .decode_final(
            &GeneratedTokens::Chunked(chunks),
            metadata.ref_rms,
            metadata.postprocess_output,
        )
        .unwrap();
    let metrics = actual.parity_metrics(&expected).unwrap();

    assert_eq!(metrics.sample_rate, 24_000);
    assert_eq!(metrics.frame_count, expected.frame_count());
    assert!(metrics.mae < 1.0e-6, "{metrics:?}");
    assert!(metrics.rmse < 1.0e-6, "{metrics:?}");
    assert!(metrics.max_abs < 1.0e-5, "{metrics:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn debug_stage1_cuda_final_matches_reference_waveform() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("debug_auto_en_short")
    {
        return;
    }
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
    assert_eq!(metrics.frame_count, reference.frame_count());
    // Measured CUDA f32 DAC drift vs GPU-only v2 oracle: max_abs ~1–2e-3 with
    // MAE/RMSE ~1e-4. Keep the same budget as other CUDA stage1 finals.
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn debug_stage1_cuda_requested_f16_does_not_collapse_to_silence() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("debug_auto_en_short")
    {
        return;
    }
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = cuda_f16_pipeline();
    let actual = pipeline
        .decode_stage1_final_from_reference_case(reference_root(), "debug_auto_en_short")
        .unwrap();

    let peak = actual
        .samples
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));

    assert!(
        peak > 1.0e-3,
        "stage1 CUDA requested f16 collapsed to silence, peak={peak}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn chunked_stage1_cuda_final_matches_reference_waveform() {
    if !support::model_fixture_available()
        || !support::reference_fixture_available("auto_long_chunked")
    {
        return;
    }
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
    assert_eq!(metrics.frame_count, reference.frame_count());
    assert!(metrics.mae < 1.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 2.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 3.0e-3, "{metrics:?}");
}
