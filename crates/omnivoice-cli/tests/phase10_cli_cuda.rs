#![cfg(feature = "cuda")]

use std::{fs, path::PathBuf, process::Command};

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
fn phase10_cli_prepare_prompt_cuda_smoke() {
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
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(stdout.contains("command=prepare-prompt"));
    assert!(stdout.contains("stage0_loaded=false"));
    assert!(stdout.contains("stage1_loaded=false"));
}

#[test]
fn phase10_cli_stage1_decode_cuda_smoke() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let out = repo_root()
        .join("artifacts")
        .join("phase10-test")
        .join("cuda_stage1_final.wav");
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
            "cuda:0",
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
    assert!(stdout.contains("stage1_loaded=true"));
}

#[test]
fn phase10_cli_stage0_debug_cuda_smoke() {
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
            "cuda:0",
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
    assert!(stdout.contains("stage0_loaded=true"));
    assert!(stdout.contains("stage1_loaded=false"));
}

#[test]
fn phase10_cli_infer_cuda_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let output_path = repo_root()
        .join("artifacts")
        .join("phase10-test")
        .join("cli_auto_en_short.wav");
    let mut args = vec![
        "infer".to_string(),
        "--model".to_string(),
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
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual = DecodedAudio::read_wav(&output_path).unwrap();
    let expected = case.load_final_audio().unwrap();
    let metrics = actual.parity_metrics(&expected).unwrap();
    assert_eq!(actual.sample_rate, expected.sample_rate);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase_marker=omnivoice-phase10"));
    assert!(metrics.mae < 5.0e-4, "{metrics:?}");
    assert!(metrics.rmse < 8.0e-4, "{metrics:?}");
    assert!(metrics.max_abs < 0.05, "{metrics:?}");
}

#[test]
fn phase10_cli_infer_cuda_auto_device_dtype_succeeds() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let output_path = repo_root()
        .join("artifacts")
        .join("phase10-test")
        .join("cli_auto_en_short_auto.wav");
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
    assert!(stdout.contains("resolved_device=Cuda(0)"));
    assert!(stdout.contains("resolved_dtype=F16"));

    let actual = DecodedAudio::read_wav(&output_path).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual, &expected, 20_000, 3.0e-2, 5.0e-2, 0.55,
    );
}

#[test]
fn phase10_cli_infer_batch_cuda_generates_expected_outputs() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();

    let batch_root = repo_root().join("artifacts").join("phase10-batch-test");
    let test_list = batch_root.join("test_list.jsonl");
    let res_dir = batch_root.join("results");
    fs::create_dir_all(&res_dir).unwrap();
    fs::write(
        &test_list,
        format!(
            "{{\"id\":\"sample_a\",\"text\":{:?}}}\n{{\"id\":\"sample_b\",\"text\":{:?}}}\n",
            request.texts[0], request.texts[0]
        ),
    )
    .unwrap();

    let output = Command::new(binary)
        .args([
            "infer-batch",
            "--model",
            &model_root().display().to_string(),
            "--test-list",
            &test_list.display().to_string(),
            "--res-dir",
            &res_dir.display().to_string(),
            "--lang-id",
            "en",
            "--batch-size",
            "2",
            "--device",
            "cuda:0",
            "--dtype",
            "f32",
            "--seed",
            "1234",
            "--num-step",
            "32",
            "--guidance-scale",
            "2.0",
            "--t-shift",
            "0.1",
            "--layer-penalty-factor",
            "5.0",
            "--position-temperature",
            "0.0",
            "--class-temperature",
            "0.0",
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
    assert!(stdout.contains("command=infer-batch"));
    assert!(stdout.contains("written_files=2"));
    assert!(stdout.contains("resolved_workers=1"));
    assert!(stdout.contains("resolved_worker_devices=Cuda(0)"));

    let expected = case.load_final_audio().unwrap();
    let actual_a = DecodedAudio::read_wav(res_dir.join("sample_a.wav")).unwrap();
    let actual_b = DecodedAudio::read_wav(res_dir.join("sample_b.wav")).unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual_a, &expected, 480, 3.0e-3, 1.0e-2, 0.4,
    );
    assert_audio_matches_reference_with_frame_tolerance(
        &actual_b, &expected, 480, 3.0e-3, 1.0e-2, 0.4,
    );
}

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
