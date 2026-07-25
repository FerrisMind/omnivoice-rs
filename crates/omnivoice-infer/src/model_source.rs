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
    if looks_like_local_path(raw_value) {
        return Err(OmniVoiceError::MissingArtifact { path: candidate });
    }
    Ok(ModelSource::HuggingFaceRepo(normalize_repo_id(raw_value)))
}

pub fn resolve_tts_model_root(model_spec: Option<&str>) -> Result<PathBuf> {
    match classify_tts_model_source(model_spec)? {
        ModelSource::LocalPath(path) => {
            ensure_local_runtime_manifest(&path)?;
            Ok(path)
        }
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
        let Some(normalized) = normalize_safe_relative_path(path) else {
            continue;
        };
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
        Err(_error) if repo_id == DEFAULT_OMNIVOICE_REPO => {
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
    if manifest_path.is_file() {
        return Ok(manifest_path);
    }

    let lock_path = snapshot_root.join(".omnivoice.artifacts.lock");
    let mut lock = LockFile::open(&lock_path)?;
    lock.lock()?;
    let write_result = if !manifest_path.is_file() {
        let _ = manifest;
        fs::write(&manifest_path, OFFICIAL_OMNIVOICE_MANIFEST_JSON.as_bytes())
    } else {
        Ok(())
    };
    let unlock_result = lock.unlock();
    write_result?;
    unlock_result?;
    Ok(manifest_path)
}

pub(crate) fn ensure_local_runtime_manifest(model_root: &Path) -> Result<PathBuf> {
    let manifest_path = model_root.join(RUNTIME_MANIFEST_FILE_NAME);
    if manifest_path.is_file() {
        return Ok(manifest_path);
    }

    let required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "audio_tokenizer/config.json",
        "audio_tokenizer/model.safetensors",
        "audio_tokenizer/preprocessor_config.json",
    ];
    if !required_files
        .iter()
        .all(|relative| model_root.join(relative).is_file())
    {
        return Err(OmniVoiceError::MissingArtifact {
            path: manifest_path,
        });
    }

    let lock_path = model_root.join(".omnivoice.artifacts.lock");
    let mut lock = LockFile::open(&lock_path)?;
    lock.lock()?;
    let write_result = if !manifest_path.is_file() {
        fs::write(&manifest_path, OFFICIAL_OMNIVOICE_MANIFEST_JSON.as_bytes())
    } else {
        Ok(())
    };
    let unlock_result = lock.unlock();
    write_result?;
    unlock_result?;
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

pub(crate) fn looks_like_local_path(raw_value: &str) -> bool {
    let path = Path::new(raw_value);
    path.is_absolute()
        || raw_value.starts_with("./")
        || raw_value.starts_with("../")
        || raw_value.starts_with(".\\")
        || raw_value.starts_with("..\\")
        || raw_value.contains('\\')
        || (raw_value.len() >= 2 && raw_value.as_bytes()[1] == b':')
}

fn normalize_safe_relative_path(path: &Path) -> Option<String> {
    let normalized = path.to_string_lossy().replace('\\', "/");
    if normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.contains(':')
        || normalized
            .split('/')
            .any(|component| component == "." || component == "..")
    {
        return None;
    }
    Some(normalized)
}

#[cfg(test)]
mod tests {
    use super::{classify_model_source, ensure_local_runtime_manifest, ModelSource};
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temporary_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("omnivoice-model-source-{name}-{nonce}"))
    }

    #[test]
    fn missing_windows_path_is_not_treated_as_huggingface_repo() {
        let error =
            classify_model_source(Some(r"C:\definitely-missing\omnivoice"), "fallback/repo")
                .unwrap_err();
        assert!(error.to_string().contains("definitely-missing"));
    }

    #[test]
    fn local_official_layout_gets_manifest_materialized() {
        let root = temporary_root("manifest");
        fs::create_dir_all(root.join("audio_tokenizer")).unwrap();
        for relative in [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "audio_tokenizer/config.json",
            "audio_tokenizer/model.safetensors",
            "audio_tokenizer/preprocessor_config.json",
        ] {
            fs::write(root.join(relative), b"fixture").unwrap();
        }

        let manifest_path = ensure_local_runtime_manifest(&root).unwrap();
        assert!(manifest_path.is_file());
        assert!(fs::read_to_string(manifest_path)
            .unwrap()
            .contains("\"version\": 1"));
        assert!(matches!(
            classify_model_source(root.to_str(), "fallback/repo").unwrap(),
            ModelSource::LocalPath(_)
        ));

        fs::remove_dir_all(root).unwrap();
    }
}
