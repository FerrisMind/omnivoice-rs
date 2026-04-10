use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(default)]
    pub voice: Option<Value>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub stream_format: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechResponseFormat {
    Wav,
    Pcm,
    Mp3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechStreamFormat {
    Audio,
    Sse,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub service: &'static str,
    pub author: &'static str,
}

fn default_response_format() -> String {
    "wav".to_string()
}
