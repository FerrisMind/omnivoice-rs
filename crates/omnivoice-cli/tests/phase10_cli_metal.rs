#![cfg(all(feature = "metal", target_os = "macos"))]

use std::{path::PathBuf, process::Command};

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle, contracts::DecodedAudio, gpu_lock::acquire_gpu_test_lock,
};

fn repo_root() -> PathBuf {
    std::env::var_os("OMNIVOICE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        })
}

fn model_root() -> PathBuf {
    repo_root().join("model")
}

fn reference_root() -> PathBuf {
    repo_root().join("artifacts").join("python_reference")
}

fn deterministic_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage7_cuda_f32_dense")
}

#[test]
fn phase10_cli_prepare_prompt_metal_smoke() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary)
        .args([
            "prepare-prompt",
            "--model-dir",
            &model_root().display().to_string(),
            "--reference-root",
            &reference_root().display().to_string(),
            "--case",
            "debug_auto_en_short",
            "--device",
            "metal",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=prepare-prompt"));
}

#[test]
fn phase10_cli_stage1_decode_metal_smoke() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let out = repo_root()
        .join("artifacts")
        .join("phase10-test")
        .join("metal_stage1_final.wav");
    let output = Command::new(binary)
        .args([
            "stage1-decode",
            "--model-dir",
            &model_root().display().to_string(),
            "--reference-root",
            &reference_root().display().to_string(),
            "--case",
            "debug_auto_en_short",
            "--out",
            &out.display().to_string(),
            "--device",
            "metal",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=stage1-decode"));
}

#[test]
fn phase10_cli_stage0_debug_metal_smoke() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary)
        .args([
            "stage0-debug",
            "--model-dir",
            &model_root().display().to_string(),
            "--reference-root",
            &reference_root().display().to_string(),
            "--case",
            "debug_auto_en_short",
            "--device",
            "metal",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=stage0-debug"));
}

#[test]
fn phase10_cli_infer_metal_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let output_path = repo_root()
        .join("artifacts")
        .join("phase10-test")
        .join("cli_auto_en_short_metal.wav");
    let mut args = vec![
        "infer".to_string(),
        "--model-dir".to_string(),
        model_root().display().to_string(),
        "--text".to_string(),
        request.texts[0].clone(),
        "--output".to_string(),
        output_path.display().to_string(),
        "--device".to_string(),
        "metal".to_string(),
        "--dtype".to_string(),
        "f32".to_string(),
        "--seed".to_string(),
        "1234".to_string(),
        "--num-step".to_string(),
        "32".to_string(),
        "--guidance-scale".to_string(),
        "2.0".to_string(),
        "--t-shift".to_string(),
        "0.1".to_string(),
        "--layer-penalty-factor".to_string(),
        "5.0".to_string(),
        "--position-temperature".to_string(),
        "0.0".to_string(),
        "--class-temperature".to_string(),
        "0.0".to_string(),
    ];
    if let Some(language) = request.languages[0].clone() {
        args.push("--language".to_string());
        args.push(language);
    }
    let output = Command::new(binary).args(&args).output().unwrap();

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual = DecodedAudio::read_wav(&output_path).unwrap();
    let expected = case.load_final_audio().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    // Metal's stage0 unmasking order diverges from the CUDA oracle after a few
    // steps even with f32, so compare the utterance envelope instead of
    // requiring sample-exact phase.
    assert_audio_perceptually_matches_reference(&actual, &expected, 480, 0.5, 6.0);
}

#[test]
fn phase10_cli_infer_metal_auto_device_dtype_succeeds() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let output_path = repo_root()
        .join("artifacts")
        .join("phase10-test")
        .join("cli_auto_en_short_auto_metal.wav");
    let mut args = vec![
        "infer".to_string(),
        "--model-dir".to_string(),
        model_root().display().to_string(),
        "--text".to_string(),
        request.texts[0].clone(),
        "--output".to_string(),
        output_path.display().to_string(),
        "--device".to_string(),
        "auto".to_string(),
        "--dtype".to_string(),
        "auto".to_string(),
        "--seed".to_string(),
        "1234".to_string(),
        "--num-step".to_string(),
        "32".to_string(),
        "--guidance-scale".to_string(),
        "2.0".to_string(),
        "--t-shift".to_string(),
        "0.1".to_string(),
        "--layer-penalty-factor".to_string(),
        "5.0".to_string(),
        "--position-temperature".to_string(),
        "0.0".to_string(),
        "--class-temperature".to_string(),
        "0.0".to_string(),
    ];
    if let Some(language) = request.languages[0].clone() {
        args.push("--language".to_string());
        args.push(language);
    }
    let output = Command::new(binary).args(&args).output().unwrap();

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=infer"));
    assert!(stdout.contains("device=Auto"));
    assert!(stdout.contains("dtype=Auto"));
    assert!(stdout.contains("resolved_device=Metal"));
    // `DTypeSpec::Auto` uses f16 stage0 activations on Metal.
    assert!(stdout.contains("resolved_dtype=F16"));

    // Auto selects f16 stage0 activations, whose unmasking trajectory differs
    // from the f32 oracle. Check that the pipeline runs and writes audio here;
    // the explicit f32 test above covers waveform parity.
    let actual = DecodedAudio::read_wav(&output_path).unwrap();
    assert_eq!(actual.sample_rate, 24_000);
    assert!(!actual.samples.is_empty());
    assert!(actual.samples.iter().all(|sample| sample.is_finite()));
}

fn assert_audio_perceptually_matches_reference(
    actual: &DecodedAudio,
    expected: &DecodedAudio,
    max_frame_delta: usize,
    min_envelope_correlation: f32,
    max_rms_ratio_db: f32,
) {
    assert_eq!(actual.sample_rate, expected.sample_rate);
    let frame_delta = actual.frame_count().abs_diff(expected.frame_count());
    assert!(
        frame_delta <= max_frame_delta,
        "frame delta {} exceeds {} (actual={}, reference={})",
        frame_delta,
        max_frame_delta,
        actual.frame_count(),
        expected.frame_count()
    );
    assert!(
        actual.samples.iter().all(|sample| sample.is_finite()),
        "actual audio contains non-finite samples"
    );

    let window = (actual.sample_rate as usize / 50).max(1);
    let correlation = audio_envelope_correlation(actual, expected, window);
    assert!(
        correlation >= min_envelope_correlation,
        "audio envelope correlation {correlation} is below {min_envelope_correlation}"
    );

    let compare_len = actual.frame_count().min(expected.frame_count());
    let actual_power = actual.samples[..compare_len]
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>();
    let expected_power = expected.samples[..compare_len]
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>();
    if actual_power > f64::EPSILON && expected_power > f64::EPSILON {
        let ratio_db = (10.0 * (actual_power / expected_power).log10()).abs();
        assert!(
            ratio_db <= f64::from(max_rms_ratio_db),
            "audio RMS ratio {ratio_db} dB exceeds {max_rms_ratio_db} dB"
        );
    }
}

fn audio_envelope_correlation(
    actual: &DecodedAudio,
    expected: &DecodedAudio,
    window: usize,
) -> f32 {
    let compare_len = actual.frame_count().min(expected.frame_count());
    let window = window.max(1);
    let mut actual_envelope = Vec::new();
    let mut expected_envelope = Vec::new();
    let mut start = 0;
    while start + window <= compare_len {
        let mut actual_power = 0.0_f64;
        let mut expected_power = 0.0_f64;
        for offset in 0..window {
            let actual_sample = f64::from(actual.samples[start + offset]);
            let expected_sample = f64::from(expected.samples[start + offset]);
            actual_power += actual_sample * actual_sample;
            expected_power += expected_sample * expected_sample;
        }
        actual_envelope.push((actual_power / window as f64).sqrt());
        expected_envelope.push((expected_power / window as f64).sqrt());
        start += window;
    }
    if actual_envelope.len() < 2 {
        return 0.0;
    }
    let count = actual_envelope.len() as f64;
    let actual_mean = actual_envelope.iter().sum::<f64>() / count;
    let expected_mean = expected_envelope.iter().sum::<f64>() / count;
    let mut covariance = 0.0_f64;
    let mut actual_variance = 0.0_f64;
    let mut expected_variance = 0.0_f64;
    for (actual_value, expected_value) in actual_envelope.iter().zip(&expected_envelope) {
        let actual_centered = actual_value - actual_mean;
        let expected_centered = expected_value - expected_mean;
        covariance += actual_centered * expected_centered;
        actual_variance += actual_centered * actual_centered;
        expected_variance += expected_centered * expected_centered;
    }
    if actual_variance <= f64::EPSILON || expected_variance <= f64::EPSILON {
        return 0.0;
    }
    (covariance / (actual_variance.sqrt() * expected_variance.sqrt())) as f32
}
