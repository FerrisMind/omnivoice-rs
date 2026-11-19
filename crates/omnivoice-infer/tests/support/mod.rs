#![allow(dead_code)]

use std::path::PathBuf;

use omnivoice_infer::contracts::{DecodedAudio, I64Tensor2};

pub fn repo_root() -> PathBuf {
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

pub fn model_root() -> PathBuf {
    repo_root().join("model")
}

pub fn reference_root() -> PathBuf {
    repo_root().join("artifacts").join("python_reference")
}

pub fn deterministic_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage7_cuda_f32_dense")
}

pub fn stage0_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage0_deterministic")
}

pub fn stage0_cpu_strict_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage0_deterministic_cpu_strict")
}

/// CPU mirror of ``python_reference_stage0_cuda_debug`` (debug det cases).
pub fn stage0_cpu_debug_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage0_cpu_debug")
}

pub fn stage0_cuda_debug_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage0_cuda_debug")
}

/// CPU mirror of the dense stage7 CUDA baseline.
pub fn stage0_cpu_dense_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage7_cpu_f32_dense")
}

pub fn ref_audio_path() -> PathBuf {
    repo_root().join("ref.wav")
}

pub fn model_fixture_available() -> bool {
    let root = model_root();
    root.join("omnivoice.artifacts.json").is_file()
        || (root.join("config.json").is_file()
            && root.join("model.safetensors").is_file()
            && root.join("tokenizer.json").is_file()
            && root.join("tokenizer_config.json").is_file()
            && root.join("audio_tokenizer/config.json").is_file()
            && root.join("audio_tokenizer/model.safetensors").is_file()
            && root
                .join("audio_tokenizer/preprocessor_config.json")
                .is_file())
}

pub fn reference_fixture_available(case_id: &str) -> bool {
    reference_root().join(case_id).join("case.json").is_file()
}

pub fn deterministic_fixture_available(case_id: &str) -> bool {
    deterministic_reference_root()
        .join(case_id)
        .join("case.json")
        .is_file()
}

pub fn live_oracle_clone_prompt() -> (I64Tensor2, String) {
    let value: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            repo_root()
                .join("artifacts")
                .join("live_oracles")
                .join("clone_prompt.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let dims = value["dims"].as_array().unwrap();
    let rows = dims[0].as_u64().unwrap() as usize;
    let cols = dims[1].as_u64().unwrap() as usize;
    let mut flat = Vec::with_capacity(rows * cols);
    for row in value["tokens"].as_array().unwrap() {
        for token in row.as_array().unwrap() {
            flat.push(token.as_i64().unwrap());
        }
    }
    (
        I64Tensor2::new((rows, cols), flat).unwrap(),
        value["ref_text"].as_str().unwrap().to_string(),
    )
}

pub fn assert_audio_matches_reference_with_frame_tolerance(
    actual: &DecodedAudio,
    expected: &DecodedAudio,
    max_frame_delta: usize,
    mae_limit: f32,
    rmse_limit: f32,
    max_abs_limit: f32,
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
    let compare_len = actual.frame_count().min(expected.frame_count());
    let actual = DecodedAudio::new(actual.samples[..compare_len].to_vec(), actual.sample_rate);
    let expected = DecodedAudio::new(
        expected.samples[..compare_len].to_vec(),
        expected.sample_rate,
    );
    let metrics = actual.parity_metrics(&expected).unwrap();
    assert!(metrics.mae < mae_limit, "{metrics:?}");
    assert!(metrics.rmse < rmse_limit, "{metrics:?}");
    assert!(metrics.max_abs < max_abs_limit, "{metrics:?}");
}

/// Pearson correlation of short-time RMS envelopes, ignoring absolute phase.
///
/// Cross-backend audio parity tests cannot require sample-exact waveforms:
/// tiny numerical differences in a convolution decoder shift high-frequency
/// phase while leaving the perceived speech and loudness contour intact.
pub fn audio_envelope_correlation(
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

/// Assert that `actual` is the same utterance and loudness as `expected`
/// without requiring sample-exact phase.
pub fn assert_audio_perceptually_matches_reference(
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
