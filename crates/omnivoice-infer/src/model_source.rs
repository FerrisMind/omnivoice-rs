use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use fslock::LockFile;
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};

use crate::{
    artifacts::{RuntimeArtifactManifest, RUNTIME_MANIFEST_FILE_NAME},
    error::{OmniVoiceError, Result},
};

pub const DEFAULT_OMNIVOICE_REPO: &str = "k2-fsa/OmniVoice";
pub const DEFAULT_WHISPER_REPO: &str = "oxide-lab/whisper-large-v3-turbo-GGUF";

const OFFICIAL_OMNIVOICE_MANIFEST_JSON: &str = include_str!("../assets/omnivoice.artifacts.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    LocalPath(PathBuf),
    HuggingFaceRepo(String),
}

pub fn classify_tts_model_source(model_spec: Option<&str>) -> Result<ModelSource> {
    classify_model_source(model_spec, DEFAULT_OMNIVOICE_REPO)
}

pub fn classify_model_source(model_spec: Option<&str>, default_repo: &str) -> Result<ModelSource> {
    let trimmed = model_spec.map(str::trim).filter(|value| !value.is_empty());
    let raw_value = trimmed.unwrap_or(default_repo);
    let candidate = PathBuf::from(raw_value);
    if candidate.exists() {
        return Ok(ModelSource::LocalPath(candidate));
    }
    Ok(ModelSource::HuggingFaceRepo(normalize_repo_id(raw_value)))
}

pub fn resolve_tts_model_root(model_spec: Option<&str>) -> Result<PathBuf> {
    match classify_tts_model_source(model_spec)? {
        ModelSource::LocalPath(path) => Ok(path),
        ModelSource::HuggingFaceRepo(repo_id) => download_tts_snapshot(&repo_id),
    }
}

pub fn resolve_tts_model_root_from_path(model_spec: Option<&Path>) -> Result<PathBuf> {
    let owned = model_spec.map(|path| path.to_string_lossy().into_owned());
    resolve_tts_model_root(owned.as_deref())
}

pub fn load_official_omnivoice_manifest() -> Result<RuntimeArtifactManifest> {
    let manifest: RuntimeArtifactManifest = serde_json::from_str(OFFICIAL_OMNIVOICE_MANIFEST_JSON)?;
    if manifest.version != 1 {
        return Err(OmniVoiceError::Unsupported(format!(
            "unsupported runtime artifact manifest version {}",
            manifest.version
        )));
    }
    Ok(manifest)
}

pub fn manifest_download_targets(manifest: &RuntimeArtifactManifest) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut targets = Vec::new();
    for path in [
        Some(manifest.generator.config.as_path()),
        Some(manifest.generator.weights.as_path()),
        Some(manifest.text_tokenizer.tokenizer.as_path()),
        Some(manifest.text_tokenizer.tokenizer_config.as_path()),
        manifest.text_tokenizer.metadata.chat_template.as_deref(),
        Some(manifest.audio_tokenizer.config.as_path()),
        Some(manifest.audio_tokenizer.weights.as_path()),
        Some(manifest.audio_tokenizer.preprocessor_config.as_path()),
        manifest.audio_tokenizer.metadata.license.as_deref(),
    ] {
        let Some(path) = path else {
            continue;
        };
        let normalized = path.to_string_lossy().replace('\\', "/");
        if normalized == RUNTIME_MANIFEST_FILE_NAME {
            continue;
        }
        if seen.insert(normalized.clone()) {
            targets.push(normalized);
        }
    }
    targets
}

impl ModelSource {
    pub fn into_spec(self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::HuggingFaceRepo(repo_id) => repo_id,
        }
    }
}

fn download_tts_snapshot(repo_id: &str) -> Result<PathBuf> {
    let api = Api::new().map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    match repo.get(RUNTIME_MANIFEST_FILE_NAME) {
        Ok(manifest_path) => {
            let snapshot_root = snapshot_root_from_manifest_path(&manifest_path)?;
            let manifest: RuntimeArtifactManifest =
                serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
            for target in manifest_download_targets(&manifest) {
                repo.get(&target)
                    .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
            }
            Ok(snapshot_root)
        }
        Err(error) if repo_id == DEFAULT_OMNIVOICE_REPO => {
            download_official_omnivoice_snapshot(&repo)
        }
        Err(error) => Err(OmniVoiceError::InvalidData(format!(
            "model repo `{repo_id}` is missing required `{RUNTIME_MANIFEST_FILE_NAME}`: {error}"
        ))),
    }
}

fn download_official_omnivoice_snapshot(repo: &ApiRepo) -> Result<PathBuf> {
    let manifest = load_official_omnivoice_manifest()?;
    let targets = manifest_download_targets(&manifest);
    let anchor = targets.first().ok_or_else(|| {
        OmniVoiceError::InvalidData(
            "official OmniVoice manifest does not define any files".to_string(),
        )
    })?;
    let anchor_path = repo
        .get(anchor)
        .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    let snapshot_root = anchor_path.parent().ok_or_else(|| {
        OmniVoiceError::InvalidData(format!(
            "hf-hub returned an invalid snapshot path for {}",
            repo.url(anchor)
        ))
    })?;

    for target in &targets {
        repo.get(target)
            .map_err(|error| OmniVoiceError::InvalidData(error.to_string()))?;
    }

    materialize_runtime_manifest(snapshot_root, &manifest)?;
    Ok(snapshot_root.to_path_buf())
}

fn materialize_runtime_manifest(
    snapshot_root: &Path,
    manifest: &RuntimeArtifactManifest,
) -> Result<PathBuf> {
    let manifest_path = snapshot_root.join(RUNTIME_MANIFEST_FILE_NAME);
    if manifest_path.exists() {
        return Ok(manifest_path);
    }

    let lock_path = snapshot_root.join(".omnivoice.artifacts.lock");
    let mut lock = LockFile::open(&lock_path)?;
    lock.lock()?;
    if !manifest_path.exists() {
        let _ = manifest;
        fs::write(&manifest_path, OFFICIAL_OMNIVOICE_MANIFEST_JSON.as_bytes())?;
    }
    let _ = lock.unlock();
    Ok(manifest_path)
}

fn snapshot_root_from_manifest_path(path: &Path) -> Result<PathBuf> {
    path.parent().map(Path::to_path_buf).ok_or_else(|| {
        OmniVoiceError::InvalidData(format!(
            "hf-hub returned an invalid snapshot path for {}",
            path.display()
        ))
    })
}

fn normalize_repo_id(raw_value: &str) -> String {
    raw_value.replace('\\', "/")
}
