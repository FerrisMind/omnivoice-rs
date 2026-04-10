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

pub fn ref_audio_path() -> PathBuf {
    repo_root().join("ref.wav")
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
