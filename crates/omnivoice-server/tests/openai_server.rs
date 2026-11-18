use std::{
    io::Cursor,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
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
    sleep_for: Mutex<Option<Duration>>,
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
                sleep_for: Mutex::new(None),
            }),
        }
    }

    fn requests(&self) -> Vec<GenerationRequest> {
        self.inner.requests.lock().unwrap().clone()
    }

    fn last_seed(&self) -> Option<u64> {
        *self.inner.last_seed.lock().unwrap()
    }

    fn set_sleep_for(&self, duration: Duration) {
        *self.inner.sleep_for.lock().unwrap() = Some(duration);
    }
}

impl SpeechRuntime for RecordingRuntime {
    fn synthesize(
        &self,
        request: GenerationRequest,
    ) -> omnivoice_infer::Result<GeneratedAudioResult> {
        if let Some(duration) = *self.inner.sleep_for.lock().unwrap() {
            std::thread::sleep(duration);
        }
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
        base_path: String::new(),
        max_body_bytes: 5 * 1024 * 1024,
        max_concurrent_requests: 1,
        mp3_bitrate_kbps: 128,
        request_timeout: Duration::from_secs(30),
    };
    let app = build_router(AppState::new(runtime.clone(), config));
    (app, runtime)
}

fn app_with_config(config: ServerConfig) -> (axum::Router, RecordingRuntime) {
    let runtime = RecordingRuntime::new();
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

fn auth_request_with_content_type(
    method: &str,
    uri: &str,
    content_type: &str,
    body: Body,
) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("Authorization", format!("Bearer {API_KEY}"))
        .header("Content-Type", content_type)
        .body(body)
        .unwrap()
}

fn options_request(uri: &str, requested_method: &str) -> Request<Body> {
    Request::builder()
        .method("OPTIONS")
        .uri(uri)
        .header(header::ORIGIN, "http://localhost:3000")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, requested_method)
        .header(
            header::ACCESS_CONTROL_REQUEST_HEADERS,
            "authorization,content-type",
        )
        .body(Body::empty())
        .unwrap()
}

fn clone_wav_bytes() -> Vec<u8> {
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
    cursor.into_inner()
}

fn clone_data_uri() -> String {
    format!(
        "data:audio/wav;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(clone_wav_bytes())
    )
}

fn multipart_body(
    boundary: &str,
    fields: &[(&str, &str)],
    file: Option<(&str, &str, &str, &[u8])>,
) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    if let Some((name, filename, content_type, bytes)) = file {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
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
async fn health_endpoint_reports_starting_state_with_503() {
    let config = ServerConfig {
        served_model_id: MODEL_ID.to_string(),
        api_key: API_KEY.to_string(),
        base_path: String::new(),
        max_body_bytes: 5 * 1024 * 1024,
        max_concurrent_requests: 1,
        mp3_bitrate_kbps: 128,
        request_timeout: Duration::from_secs(30),
    };
    let app = build_router(AppState::starting(config));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn audio_speech_returns_503_while_runtime_is_starting() {
    let config = ServerConfig {
        served_model_id: MODEL_ID.to_string(),
        api_key: API_KEY.to_string(),
        base_path: String::new(),
        max_body_bytes: 5 * 1024 * 1024,
        max_concurrent_requests: 1,
        mp3_bitrate_kbps: 128,
        request_timeout: Duration::from_secs(30),
    };
    let app = build_router(AppState::starting(config));

    let response = app
        .oneshot(auth_request(
            "POST",
            "/v1/audio/speech",
            Body::from(
                json!({
                    "model": MODEL_ID,
                    "input": "hello",
                    "voice": "alloy"
                })
                .to_string(),
            ),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
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
async fn routes_respect_base_path() {
    let config = ServerConfig {
        served_model_id: MODEL_ID.to_string(),
        api_key: API_KEY.to_string(),
        base_path: "/edge".to_string(),
        max_body_bytes: 5 * 1024 * 1024,
        max_concurrent_requests: 1,
        mp3_bitrate_kbps: 128,
        request_timeout: Duration::from_secs(30),
    };
    let (app, _) = app_with_config(config);

    let prefixed = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/edge/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(prefixed.status(), StatusCode::OK);

    let unprefixed = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unprefixed.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn options_preflight_is_available_for_public_routes() {
    let (app, _) = app_with_runtime();
    for (path, method) in [
        ("/", "GET"),
        ("/health", "GET"),
        ("/v1/models", "GET"),
        ("/v1/audio/speech", "POST"),
    ] {
        let response = app
            .clone()
            .oneshot(options_request(path, method))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "failed for {path}");
        assert!(response
            .headers()
            .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));
    }
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
                    "asr_language": "hi",
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
    assert_eq!(recorded[0].asr_language.as_deref(), Some("hi"));
    assert_eq!(recorded[0].speeds[0], Some(1.25));
    assert!(!recorded[0].generation_config.denoise);
    assert_eq!(recorded[0].generation_config.num_step, 16);
    assert!(matches!(
        recorded[0].ref_audios[0],
        Some(ReferenceAudioInput::Waveform(_))
    ));
}

#[tokio::test]
async fn multipart_clone_request_maps_extensions_into_generation_request() {
    let (app, runtime) = app_with_runtime();
    let boundary = "omnivoice-boundary";
    let body = multipart_body(
        boundary,
        &[
            ("model", MODEL_ID),
            ("input", "clone me"),
            ("voice", "alloy"),
            ("instructions", "base instructions"),
            ("response_format", "wav"),
            ("speed", "1.25"),
            ("language", "en"),
            ("duration", "2.5"),
            ("ref_text", "reference text"),
            ("asr_language", "hi"),
            ("instruct", "preferred instruct"),
            ("seed", "42"),
            ("num_step", "16"),
            ("denoise", "false"),
        ],
        Some(("ref_audio", "ref.wav", "audio/wav", &clone_wav_bytes())),
    );

    let response = app
        .oneshot(auth_request_with_content_type(
            "POST",
            "/v1/audio/speech",
            &format!("multipart/form-data; boundary={boundary}"),
            Body::from(body),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
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
    assert_eq!(recorded[0].asr_language.as_deref(), Some("hi"));
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
async fn request_timeout_returns_request_timeout_status() {
    let (_, runtime) = app_with_runtime();
    runtime.set_sleep_for(Duration::from_millis(200));

    let mut config = ServerConfig {
        served_model_id: MODEL_ID.to_string(),
        api_key: API_KEY.to_string(),
        base_path: String::new(),
        max_body_bytes: 5 * 1024 * 1024,
        max_concurrent_requests: 1,
        mp3_bitrate_kbps: 128,
        request_timeout: Duration::from_millis(25),
    };
    let app = build_router(AppState::new(runtime, config.clone()));
    config.request_timeout = Duration::from_millis(25);

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

    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
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
