use axum::{
    extract::DefaultBodyLimit,
    extract::{Request, State},
    http::{header, HeaderMap},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::{atomic::AtomicBool, Arc};
use tower_http::cors::{Any, CorsLayer};

use crate::{
    audio::{build_audio_response, parse_http_speech_request},
    error::ServerError,
    openai::{HealthResponse, ModelObject, ModelsResponse},
    runtime::{AppState, RuntimeStatus},
};

pub fn build_router(state: AppState) -> Router {
    let max_body_bytes = state.config.max_body_bytes;
    let base_path = state.config.base_path.clone();
    let api = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/audio/speech", post(audio_speech))
        .layer(DefaultBodyLimit::max(max_body_bytes));
    let router = if base_path.is_empty() {
        api
    } else {
        Router::new().nest(&base_path, api)
    };
    router.layer(cors_layer()).with_state(state)
}

async fn root() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        service: "omnivoice-server",
        author: "FerrisMind",
    })
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let (status_code, status) = match state.status() {
        RuntimeStatus::Ready => (axum::http::StatusCode::OK, "ok"),
        RuntimeStatus::Starting => (axum::http::StatusCode::SERVICE_UNAVAILABLE, "starting"),
        RuntimeStatus::Failed => (axum::http::StatusCode::SERVICE_UNAVAILABLE, "failed"),
    };
    (
        status_code,
        Json(HealthResponse {
            status: status.to_string(),
            service: "omnivoice-server",
            author: "FerrisMind",
        }),
    )
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
    request: Request,
) -> Result<impl IntoResponse, ServerError> {
    authorize(request.headers(), &state)?;
    let parsed = parse_http_speech_request(request, &state).await?;
    let response_format = parsed.response_format;
    let stream_format = parsed.stream_format;
    let seed_override = parsed.seed_override;
    let generation_request = parsed.generation_request;
    let runtime = state
        .runtime()
        .ok_or_else(|| ServerError::service_unavailable("runtime is not ready"))?;
    let request_timeout = state.config.request_timeout;
    let cancellation = Arc::new(AtomicBool::new(false));
    let cancellation_for_worker = Arc::clone(&cancellation);
    let limiter = Arc::clone(&state.limiter);
    let handle = async move {
        let permit = limiter
            .acquire_owned()
            .await
            .map_err(|_| ServerError::internal("request limiter is closed"))?;
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            runtime.synthesize_with_seed(generation_request, seed_override, cancellation_for_worker)
        })
        .await
        .map_err(ServerError::from)?
        .map_err(ServerError::from)
    };
    let result = match tokio::time::timeout(request_timeout, handle).await {
        Ok(result) => result?,
        Err(_) => {
            cancellation.store(true, std::sync::atomic::Ordering::Release);
            return Err(ServerError::request_timeout("request timed out"));
        }
    };

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

fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE])
}
