#![allow(dead_code)]

use std::{path::PathBuf, process::Command};

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn stage0_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage0_deterministic")
}

#[cfg(feature = "cuda")]
fn stage0_cpu_strict_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage0_deterministic_cpu_strict")
}

#[cfg(feature = "cuda")]
fn deterministic_reference_root() -> PathBuf {
    repo_root()
        .join("artifacts")
        .join("python_reference_stage7_cuda_f32_dense")
}

#[test]
fn artifacts_validate_smoke_test() {
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary)
        .args([
            "artifacts",
            "validate",
            "--model-dir",
            &model_root().display().to_string(),
            "--reference-root",
            &reference_root().display().to_string(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("generator_weights="));
    assert!(stdout.contains("audio_tokenizer_weights="));
    assert!(stdout.contains("reference_case_count="));
}

#[test]
fn cli_usage_prefers_gpu_first_devices() {
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary).output().unwrap();

    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--device auto|cuda|cuda:N|mps|metal|cpu"));
    assert!(stderr.contains("--dtype auto|f16|bf16|f32"));
    assert!(stderr.contains("infer-batch"));
}

#[cfg(feature = "cuda")]
#[test]
fn prepare_prompt_smoke_test() {
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
            "cuda:0",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=prepare-prompt"));
    assert!(stdout.contains("input_ids_dims="));
    assert!(stdout.contains("tokens_init_dims="));
}

#[cfg(feature = "cuda")]
#[test]
fn stage1_prepare_smoke_test() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary)
        .args([
            "stage1-prepare",
            "--model-dir",
            &model_root().display().to_string(),
            "--reference-root",
            &reference_root().display().to_string(),
            "--case",
            "debug_auto_en_short",
            "--device",
            "cuda:0",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=stage1-prepare"));
    assert!(stdout.contains("token_dims="));
    assert!(stdout.contains("sample_rate=24000"));
}

#[cfg(feature = "cuda")]
#[test]
fn stage1_decode_raw_smoke_test() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary)
        .args([
            "stage1-decode",
            "--model-dir",
            &model_root().display().to_string(),
            "--reference-root",
            &reference_root().display().to_string(),
            "--case",
            "debug_auto_en_short",
            "--raw",
            "--out",
            &repo_root()
                .join("artifacts")
                .join("phase5-test")
                .join("debug_raw.wav")
                .display()
                .to_string(),
            "--device",
            "cuda:0",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=stage1-decode"));
    assert!(stdout.contains("mode=raw"));
    assert!(stdout.contains("sample_rate=24000"));
    assert!(stdout.contains("frame_count=90240"));
    assert!(stdout.contains("max_abs="));
    assert!(stdout.contains("mae="));
    assert!(stdout.contains("rmse="));
}

#[cfg(feature = "cuda")]
#[test]
fn stage1_decode_final_smoke_test() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
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
            &repo_root()
                .join("artifacts")
                .join("phase5-test")
                .join("debug_final.wav")
                .display()
                .to_string(),
            "--device",
            "cuda:0",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=stage1-decode"));
    assert!(stdout.contains("mode=final"));
    assert!(stdout.contains("sample_rate=24000"));
    assert!(stdout.contains("frame_count=77040"));
    assert!(stdout.contains("max_abs="));
    assert!(stdout.contains("mae="));
    assert!(stdout.contains("rmse="));
}

#[cfg(feature = "cuda")]
#[test]
fn stage0_generate_smoke_test() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary)
        .args([
            "stage0-generate",
            "--model-dir",
            &model_root().display().to_string(),
            "--reference-root",
            &stage0_reference_root().display().to_string(),
            "--case",
            "det_auto_en_short",
            "--out",
            &repo_root()
                .join("artifacts")
                .join("phase6-test")
                .join("det_auto_en_short.json")
                .display()
                .to_string(),
            "--device",
            "cuda:0",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=stage0-generate"));
    assert!(stdout.contains("kind=single"));
    assert!(stdout.contains("token_dims="));
}

#[cfg(feature = "cuda")]
#[test]
fn stage0_debug_smoke_test() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary)
        .args([
            "stage0-debug",
            "--model-dir",
            &model_root().display().to_string(),
            "--reference-root",
            &stage0_cpu_strict_reference_root().display().to_string(),
            "--case",
            "det_debug_auto_en_short",
            "--device",
            "cuda:0",
            "--dtype",
            "f32",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=stage0-debug"));
    assert!(stdout.contains("metric.inputs_embeds="));
    assert!(stdout.contains("metric.final_hidden="));
    assert!(stdout.contains("metric.step_00_pred_tokens="));
}

#[cfg(feature = "cuda")]
#[test]
fn infer_cuda_auto_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let output_path = repo_root()
        .join("artifacts")
        .join("phase7-test")
        .join("cli_auto_en_short.wav");
    let mut args = vec![
        "infer".to_string(),
        "--model-dir".to_string(),
        model_root().display().to_string(),
        "--text".to_string(),
        request.texts[0].clone(),
        "--output".to_string(),
        output_path.display().to_string(),
        "--device".to_string(),
        "cuda:0".to_string(),
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
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let actual = DecodedAudio::read_wav(output_path).unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual, &expected, 480, 5.0e-4, 8.0e-4, 0.05,
    );
}

#[cfg(feature = "cuda")]
fn assert_audio_matches_reference_with_frame_tolerance(
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
