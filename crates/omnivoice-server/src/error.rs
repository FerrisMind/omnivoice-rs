use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use omnivoice_infer::OmniVoiceError;
use serde::Serialize;

#[derive(Debug)]
pub struct ServerError {
    status: StatusCode,
    error_type: &'static str,
    message: String,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    error_type: &'a str,
    param: Option<&'a str>,
    code: Option<&'a str>,
}

impl ServerError {
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "authentication_error", message)
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            message,
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }

    pub fn from_infer(error: OmniVoiceError) -> Self {
        match error {
            OmniVoiceError::InvalidRequest(message)
            | OmniVoiceError::InvalidData(message)
            | OmniVoiceError::Unsupported(message) => Self::validation(message),
            other => Self::internal(other.to_string()),
        }
    }

    fn new(status: StatusCode, error_type: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            error_type,
            message: message.into(),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let body = ErrorEnvelope {
            error: ErrorBody {
                message: &self.message,
                error_type: self.error_type,
                param: None,
                code: None,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

impl From<std::io::Error> for ServerError {
    fn from(value: std::io::Error) -> Self {
        Self::internal(value.to_string())
    }
}

impl From<OmniVoiceError> for ServerError {
    fn from(value: OmniVoiceError) -> Self {
        Self::from_infer(value)
    }
}

impl From<tokio::task::JoinError> for ServerError {
    fn from(value: tokio::task::JoinError) -> Self {
        Self::internal(value.to_string())
    }
}
