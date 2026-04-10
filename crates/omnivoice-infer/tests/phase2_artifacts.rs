mod support;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{safetensors::save, Device, Tensor};
use omnivoice_infer::{
    artifacts::{ReferenceArtifactBundle, RuntimeArtifacts},
    frontend::Frontend,
};
use support::model_root;

#[test]
fn runtime_manifest_matches_real_model_layout() {
    let artifacts = RuntimeArtifacts::from_model_root(model_root()).unwrap();

    assert_eq!(artifacts.manifest().version, 1);
    assert_eq!(artifacts.contracts().num_audio_codebooks, 8);
    assert_eq!(artifacts.contracts().audio_vocab_size, 1025);
    assert_eq!(artifacts.contracts().audio_mask_id, 1024);
    assert_eq!(artifacts.contracts().sample_rate, 24_000);
    assert_eq!(artifacts.contracts().hop_length, 960);
    assert_eq!(artifacts.contracts().frame_rate, 25);
    assert_eq!(artifacts.contracts().token_id_range(), 0..=1023);
    assert!(artifacts
        .generator()
        .weights_path()
        .ends_with("model/model.safetensors"));
    assert!(artifacts
        .audio_tokenizer()
        .weights_path()
        .ends_with("model/audio_tokenizer/model.safetensors"));
}

#[test]
fn frontend_uses_validated_runtime_artifacts() {
    let artifacts = RuntimeArtifacts::from_model_root(model_root()).unwrap();
    let frontend = Frontend::from_runtime_artifacts(&artifacts).unwrap();

    assert_eq!(frontend.frame_rate(), 25);
}

#[test]
fn reference_bundle_loads_without_runtime_manifest() {
    let temp = TestDir::new("reference-only");
    let case_dir = temp.path().join("case-a");
    fs::create_dir_all(&case_dir).unwrap();
    fs::write(
        case_dir.join("case.json"),
        r#"{"id":"case-a","kind":"debug"}"#,
    )
    .unwrap();

    let bundle = ReferenceArtifactBundle::from_root(temp.path()).unwrap();
    assert_eq!(
        bundle.available_case_ids().unwrap(),
        vec!["case-a".to_string()]
    );
}

#[test]
fn runtime_loader_rejects_missing_required_file() {
    let temp = TestDir::new("missing-generator-weights");
    write_manifest(&temp, true);
    write_runtime_support_files(&temp, false, false);

    let error = RuntimeArtifacts::from_model_root(temp.path()).unwrap_err();
    let message = error.to_string();

    assert!(message.contains("missing artifact"));
    assert!(message.contains("model.safetensors"));
}

#[test]
fn runtime_loader_allows_missing_optional_chat_template() {
    let temp = TestDir::new("optional-chat-template");
    write_manifest(&temp, true);
    write_runtime_support_files(&temp, true, false);
    write_generator_weights(
        temp.path().join("model.safetensors"),
        &["llm", "audio_embeddings", "audio_heads"],
    );
    write_audio_tokenizer_weights(
        temp.path()
            .join("audio_tokenizer")
            .join("model.safetensors"),
        &[
            "semantic_model",
            "acoustic_encoder",
            "acoustic_decoder",
            "quantizer",
        ],
    );

    let artifacts = RuntimeArtifacts::from_model_root(temp.path()).unwrap();

    assert!(artifacts.text_tokenizer().chat_template_path().is_none());
}

#[test]
fn runtime_loader_rejects_wrong_generator_prefixes() {
    let temp = TestDir::new("wrong-generator-prefixes");
    write_manifest(&temp, false);
    write_runtime_support_files(&temp, true, false);
    write_generator_weights(temp.path().join("model.safetensors"), &["bad_prefix"]);
    write_audio_tokenizer_weights(
        temp.path()
            .join("audio_tokenizer")
            .join("model.safetensors"),
        &[
            "semantic_model",
            "acoustic_encoder",
            "acoustic_decoder",
            "quantizer",
        ],
    );

    let error = RuntimeArtifacts::from_model_root(temp.path()).unwrap_err();
    let message = error.to_string();

    assert!(message.contains("generator"));
    assert!(message.contains("llm"));
}

#[test]
fn runtime_loader_rejects_wrong_audio_tokenizer_prefixes() {
    let temp = TestDir::new("wrong-audio-prefixes");
    write_manifest(&temp, false);
    write_runtime_support_files(&temp, true, false);
    write_generator_weights(
        temp.path().join("model.safetensors"),
        &["llm", "audio_embeddings", "audio_heads"],
    );
    write_audio_tokenizer_weights(
        temp.path()
            .join("audio_tokenizer")
            .join("model.safetensors"),
        &["semantic_model", "acoustic_encoder"],
    );

    let error = RuntimeArtifacts::from_model_root(temp.path()).unwrap_err();
    let message = error.to_string();

    assert!(message.contains("audio_tokenizer"));
    assert!(message.contains("acoustic_decoder"));
}

fn write_manifest(temp: &TestDir, include_chat_template: bool) {
    let chat_template_line = if include_chat_template {
        r#","metadata":{"chat_template":"chat_template.jinja"}"#
    } else {
        ""
    };

    let manifest = format!(
        r#"{{
  "version": 1,
  "generator": {{
    "config": "config.json",
    "weights": "model.safetensors",
    "required_prefixes": ["llm", "audio_embeddings", "audio_heads"],
    "ignored_keys": ["codebook_layer_offsets"]
  }},
  "text_tokenizer": {{
    "tokenizer": "tokenizer.json",
    "tokenizer_config": "tokenizer_config.json"
    {chat_template_line}
  }},
  "audio_tokenizer": {{
    "config": "audio_tokenizer/config.json",
    "weights": "audio_tokenizer/model.safetensors",
    "preprocessor_config": "audio_tokenizer/preprocessor_config.json",
    "required_prefixes": ["semantic_model", "acoustic_encoder", "acoustic_decoder", "quantizer"],
    "metadata": {{
      "license": "audio_tokenizer/LICENSE"
    }}
  }},
  "contracts": {{
    "num_audio_codebooks": 8,
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "token_id_min": 0,
    "token_id_max": 1023,
    "sample_rate": 24000,
    "hop_length": 960,
    "frame_rate": 25
  }}
}}"#
    );

    fs::write(temp.path().join("omnivoice.artifacts.json"), manifest).unwrap();
}

fn write_runtime_support_files(temp: &TestDir, include_license: bool, include_chat_template: bool) {
    fs::create_dir_all(temp.path().join("audio_tokenizer")).unwrap();
    fs::write(
        temp.path().join("config.json"),
        r#"{
  "audio_vocab_size": 1025,
  "audio_mask_id": 1024,
  "num_audio_codebook": 8,
  "llm_config": {
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 40960,
    "rms_norm_eps": 0.000001,
    "rope_parameters": {
      "rope_theta": 1000000.0
    },
    "vocab_size": 151676
  }
}"#,
    )
    .unwrap();
    fs::write(temp.path().join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(
        temp.path().join("tokenizer_config.json"),
        r#"{"model_max_length":131072}"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("audio_tokenizer").join("config.json"),
        r#"{
  "sample_rate": 24000,
  "downsample_factor": 320,
  "n_codebooks": 9,
  "acoustic_model_config": {
    "hop_length": 960
  }
}"#,
    )
    .unwrap();
    fs::write(
        temp.path()
            .join("audio_tokenizer")
            .join("preprocessor_config.json"),
        r#"{
  "sampling_rate": 24000,
  "hop_length": 960
}"#,
    )
    .unwrap();
    if include_license {
        fs::write(
            temp.path().join("audio_tokenizer").join("LICENSE"),
            "test license",
        )
        .unwrap();
    }
    if include_chat_template {
        fs::write(temp.path().join("chat_template.jinja"), "{{ message }}").unwrap();
    }
}

fn write_generator_weights(path: PathBuf, prefixes: &[&str]) {
    write_prefix_weights(path, prefixes);
}

fn write_audio_tokenizer_weights(path: PathBuf, prefixes: &[&str]) {
    write_prefix_weights(path, prefixes);
}

fn write_prefix_weights(path: PathBuf, prefixes: &[&str]) {
    let mut tensors = HashMap::new();
    for prefix in prefixes {
        let tensor = Tensor::new(vec![0f32], &Device::Cpu).unwrap();
        tensors.insert(format!("{prefix}.weight"), tensor);
    }
    save(&tensors, path).unwrap();
}

#[derive(Debug)]
struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "omnivoice-phase2-{label}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
