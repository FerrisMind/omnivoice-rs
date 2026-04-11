use std::{fs, path::PathBuf, process::Command};

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

#[test]
fn wrapper_help_commands_exit_successfully() {
    for (binary, expected_name) in [
        (env!("CARGO_BIN_EXE_omnivoice-infer"), "omnivoice-infer"),
        (
            env!("CARGO_BIN_EXE_omnivoice-infer-batch"),
            "omnivoice-infer-batch",
        ),
    ] {
        let output = Command::new(binary).arg("--help").output().unwrap();

        assert!(
            output.status.success(),
            "stdout:\n{}\n\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("usage:"));
        assert!(stdout.contains(expected_name));
    }
}

#[test]
fn infer_subcommand_help_exits_successfully() {
    let binary = env!("CARGO_BIN_EXE_omnivoice-cli");
    let output = Command::new(binary)
        .args(["infer", "--help"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("omnivoice-cli infer"));
    assert!(stdout.contains("--guidance_scale"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn official_infer_aliases_reach_runtime_validation_without_parse_errors() {
    let binary = env!("CARGO_BIN_EXE_omnivoice-infer");
    let output_path = repo_root()
        .join("artifacts")
        .join("compat-test")
        .join("official-alias.wav");
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    let output = Command::new(binary)
        .args([
            "--model",
            &model_root().display().to_string(),
            "--text",
            "hello from official alias coverage",
            "--output",
            &output_path.display().to_string(),
            "--ref_audio",
            &repo_root().join("ref.wav").display().to_string(),
            "--ref_text",
            "reference transcript",
            "--num_step",
            "1",
            "--guidance_scale",
            "2.0",
            "--t_shift",
            "0.1",
            "--postprocess_output",
            "true",
            "--layer_penalty_factor",
            "5.0",
            "--position_temperature",
            "0.0",
            "--class_temperature",
            "0.0",
            "--device",
            "cuda",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("unknown argument"), "stderr:\n{stderr}");
    assert!(
        !stderr.contains("unsupported device spec"),
        "stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("requires the `cuda` feature"),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        stderr
    );
}
