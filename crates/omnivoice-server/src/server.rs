use axum::{
    extract::DefaultBodyLimit,
    extract::{Json, State},
    http::{header, HeaderMap},
    response::IntoResponse,
    routing::{get, post},
    Router,
};

use crate::{
    audio::{build_audio_response, parse_speech_request},
    error::ServerError,
    openai::{HealthResponse, ModelObject, ModelsResponse, SpeechRequest},
    runtime::AppState,
};

pub fn build_router(state: AppState) -> Router {
    let max_body_bytes = state.config.max_body_bytes;
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/audio/speech", post(audio_speech))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

async fn root() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        service: "omnivoice-server",
        author: "FerrisMind",
    })
}

async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        service: "omnivoice-server",
        author: "FerrisMind",
    })
}

async fn models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ServerError> {
    authorize(&headers, &state)?;
    Ok(Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.config.served_model_id.clone(),
            object: "model",
            created: 0,
            owned_by: "FerrisMind",
        }],
    }))
}

async fn audio_speech(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<SpeechRequest>,
) -> Result<impl IntoResponse, ServerError> {
    authorize(&headers, &state)?;
    let parsed = parse_speech_request(request, &state.config.served_model_id)?;
    let response_format = parsed.response_format;
    let stream_format = parsed.stream_format;
    let seed_override = parsed.seed_override;
    let generation_request = parsed.generation_request;
    let runtime = state.runtime.clone();
    let permit = state
        .limiter
        .acquire()
        .await
        .map_err(|_| ServerError::internal("request limiter is closed"))?;

    let result = tokio::task::spawn_blocking(move || {
        if let Some(seed) = seed_override {
            runtime.set_seed(seed)?;
        }
        runtime.synthesize(generation_request)
    })
    .await??;
    drop(permit);

    build_audio_response(
        result,
        response_format,
        stream_format,
        state.config.mp3_bitrate_kbps,
    )
}

fn authorize(headers: &HeaderMap, state: &AppState) -> Result<(), ServerError> {
    let header = headers
        .get(header::AUTHORIZATION)
        .ok_or_else(|| ServerError::unauthorized("missing Authorization header"))?;
    let value = header
        .to_str()
        .map_err(|_| ServerError::unauthorized("Authorization header is not valid ASCII"))?;
    let Some(token) = value.strip_prefix("Bearer ") else {
        return Err(ServerError::unauthorized(
            "Authorization header must use Bearer authentication",
        ));
    };
    if token != state.config.api_key {
        return Err(ServerError::unauthorized("invalid API key"));
    }
    Ok(())
}
