use std::{
    io::Cursor,
    sync::{Arc, Mutex},
};

use axum::{body::Body, http::Request};
use base64::Engine as _;
use http_body_util::BodyExt;
use omnivoice_infer::{
    contracts::{DecodedAudio, GenerationRequest, GenerationUsage, ReferenceAudioInput},
    GeneratedAudioResult,
};
use omnivoice_server::{
    build_router,
    runtime::{AppState, ServerConfig, SpeechRuntime},
};
use serde_json::json;
use tower::ServiceExt;

const API_KEY: &str = "test-secret";
const MODEL_ID: &str = "omnivoice-tts";

#[derive(Clone)]
struct RecordingRuntime {
    inner: Arc<RecordingRuntimeInner>,
}

struct RecordingRuntimeInner {
    requests: Mutex<Vec<GenerationRequest>>,
    last_seed: Mutex<Option<u64>>,
    response: GeneratedAudioResult,
}

impl RecordingRuntime {
    fn new() -> Self {
        Self {
            inner: Arc::new(RecordingRuntimeInner {
                requests: Mutex::new(Vec::new()),
                last_seed: Mutex::new(None),
                response: GeneratedAudioResult {
                    audio: sample_audio(),
                    usage: GenerationUsage::new(7, 12),
                },
            }),
        }
    }

    fn requests(&self) -> Vec<GenerationRequest> {
        self.inner.requests.lock().unwrap().clone()
    }

    fn last_seed(&self) -> Option<u64> {
        *self.inner.last_seed.lock().unwrap()
    }
}

impl SpeechRuntime for RecordingRuntime {
    fn synthesize(
        &self,
        request: GenerationRequest,
    ) -> omnivoice_infer::Result<GeneratedAudioResult> {
        self.inner.requests.lock().unwrap().push(request);
        Ok(self.inner.response.clone())
    }

    fn set_seed(&self, seed: u64) -> omnivoice_infer::Result<()> {
        *self.inner.last_seed.lock().unwrap() = Some(seed);
        Ok(())
    }
}

fn app_with_runtime() -> (axum::Router, RecordingRuntime) {
    let runtime = RecordingRuntime::new();
    let config = ServerConfig {
        served_model_id: MODEL_ID.to_string(),
        api_key: API_KEY.to_string(),
        max_body_bytes: 5 * 1024 * 1024,
        max_concurrent_requests: 1,
        mp3_bitrate_kbps: 128,
    };
    let app = build_router(AppState::new(runtime.clone(), config));
    (app, runtime)
}

fn sample_audio() -> DecodedAudio {
    let samples = (0..480)
        .map(|index| ((index as f32 / 12.0).sin() * 0.15).clamp(-1.0, 1.0))
        .collect();
    DecodedAudio::new(samples, 24_000)
}

fn auth_request(method: &str, uri: &str, body: Body) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("Authorization", format!("Bearer {API_KEY}"))
        .header("Content-Type", "application/json")
        .body(body)
        .unwrap()
}

fn clone_data_uri() -> String {
    let mut cursor = Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for index in 0..24_000 {
            let sample = (((index as f32 / 8.0).sin() * i16::MAX as f32) * 0.1) as i16;
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }
    format!(
        "data:audio/wav;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(cursor.into_inner())
    )
}

#[tokio::test]
async fn health_endpoint_is_public() {
    let (app, _) = app_with_runtime();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(payload["author"], "FerrisMind");
}

#[tokio::test]
async fn models_endpoint_returns_openai_shape() {
    let (app, _) = app_with_runtime();
    let response = app
        .oneshot(auth_request("GET", "/v1/models", Body::empty()))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(payload["object"], "list");
    assert_eq!(payload["data"][0]["id"], MODEL_ID);
    assert_eq!(payload["data"][0]["owned_by"], "FerrisMind");
}

#[tokio::test]
async fn models_endpoint_requires_auth() {
    let (app, runtime) = app_with_runtime();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 401);
    assert!(runtime.requests().is_empty());
}

#[tokio::test]
async fn audio_speech_returns_wav() {
    let (app, _) = app_with_runtime();
    let response = app
        .oneshot(auth_request(
            "POST",
            "/v1/audio/speech",
            Body::from(
                json!({
                    "model": MODEL_ID,
                    "input": "hello",
                    "voice": "alloy",
                    "response_format": "wav"
                })
                .to_string(),
            ),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert!(response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("audio/wav"));
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let reader = hound::WavReader::new(Cursor::new(body.to_vec())).unwrap();
    assert_eq!(reader.spec().sample_rate, 24_000);
}

#[tokio::test]
async fn audio_speech_returns_pcm() {
    let (app, _) = app_with_runtime();
    let response = app
        .oneshot(auth_request(
            "POST",
            "/v1/audio/speech",
            Body::from(
                json!({
                    "model": MODEL_ID,
                    "input": "hello",
                    "voice": "alloy",
                    "response_format": "pcm"
                })
                .to_string(),
            ),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers().get("content-type").unwrap(), "audio/pcm");
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body.len(), sample_audio().samples.len() * 2);
}

#[tokio::test]
async fn audio_speech_returns_mp3() {
    let (app, _) = app_with_runtime();
    let response = app
        .oneshot(auth_request(
            "POST",
            "/v1/audio/speech",
            Body::from(
                json!({
                    "model": MODEL_ID,
                    "input": "hello",
                    "voice": "alloy",
                    "response_format": "mp3"
                })
                .to_string(),
            ),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "audio/mpeg"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert!(!body.is_empty());
}

#[tokio::test]
async fn clone_request_maps_extensions_into_generation_request() {
    let (app, runtime) = app_with_runtime();
    let response = app
        .oneshot(auth_request(
            "POST",
            "/v1/audio/speech",
            Body::from(
                json!({
                    "model": MODEL_ID,
                    "input": "clone me",
                    "voice": { "id": "voice_123" },
                    "instructions": "base instructions",
                    "response_format": "wav",
                    "speed": 1.25,
                    "language": "en",
                    "duration": 2.5,
                    "ref_text": "reference text",
                    "ref_audio": clone_data_uri(),
                    "instruct": "preferred instruct",
                    "seed": 42,
                    "num_step": 16,
                    "denoise": false
                })
                .to_string(),
            ),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let recorded = runtime.requests();
    assert_eq!(recorded.len(), 1);
    assert_eq!(runtime.last_seed(), Some(42));
    assert_eq!(recorded[0].languages[0].as_deref(), Some("en"));
    assert_eq!(recorded[0].durations[0], Some(2.5));
    assert_eq!(
        recorded[0].instructs[0].as_deref(),
        Some("preferred instruct")
    );
    assert_eq!(recorded[0].ref_texts[0].as_deref(), Some("reference text"));
    assert_eq!(recorded[0].speeds[0], Some(1.25));
    assert!(!recorded[0].generation_config.denoise);
    assert_eq!(recorded[0].generation_config.num_step, 16);
    assert!(matches!(
        recorded[0].ref_audios[0],
        Some(ReferenceAudioInput::Waveform(_))
    ));
}

#[tokio::test]
async fn invalid_model_returns_422() {
    let (app, _) = app_with_runtime();
    let response = app
        .oneshot(auth_request(
            "POST",
            "/v1/audio/speech",
            Body::from(
                json!({
                    "model": "wrong-model",
                    "input": "hello",
                    "voice": "alloy"
                })
                .to_string(),
            ),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 422);
}

#[tokio::test]
async fn invalid_ref_audio_returns_422() {
    let (app, _) = app_with_runtime();
    let response = app
        .oneshot(auth_request(
            "POST",
            "/v1/audio/speech",
            Body::from(
                json!({
                    "model": MODEL_ID,
                    "input": "hello",
                    "voice": "alloy",
                    "ref_audio": "data:audio/wav;base64,not-base64!"
                })
                .to_string(),
            ),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 422);
}

#[tokio::test]
async fn sse_stream_contains_delta_done_and_done_marker() {
    let (app, _) = app_with_runtime();
    let response = app
        .oneshot(auth_request(
            "POST",
            "/v1/audio/speech",
            Body::from(
                json!({
                    "model": MODEL_ID,
                    "input": "hello",
                    "voice": "alloy",
                    "response_format": "pcm",
                    "stream_format": "sse"
                })
                .to_string(),
            ),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert!(response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("\"type\":\"speech.audio.delta\""));
    assert!(text.contains("\"type\":\"speech.audio.done\""));
    assert!(text.contains("data: [DONE]"));
}
