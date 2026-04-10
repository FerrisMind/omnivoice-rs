#![cfg(all(feature = "metal", target_os = "macos"))]

use std::{path::PathBuf, process::Command};

use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle, contracts::DecodedAudio, gpu_lock::acquire_gpu_test_lock,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
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
            &repo_root()
                .join("artifacts")
                .join("python_reference_stage0_deterministic_cpu_strict")
                .display()
                .to_string(),
            "--case",
            "det_debug_auto_en_short",
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
    let metrics = actual.parity_metrics(&expected).unwrap();
    assert_eq!(actual.sample_rate, expected.sample_rate);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(metrics.mae < 2.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 3.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 5.0e-3, "{metrics:?}");
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
    assert!(stdout.contains("resolved_dtype=F16"));

    let actual = DecodedAudio::read_wav(&output_path).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual, &expected, 20_000, 3.0e-2, 5.0e-2, 0.55,
    );
}
