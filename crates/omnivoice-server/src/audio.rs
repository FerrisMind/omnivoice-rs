use std::{collections::HashMap, convert::Infallible, io::Cursor, path::Path};

use axum::{
    body::{Body, Bytes},
    extract::{FromRequest, Multipart, Request},
    http::{header, HeaderValue, Response, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::{stream, StreamExt};
use omnivoice_infer::{
    audio_input::load_audio_bytes,
    contracts::{DecodedAudio, GenerationConfig, GenerationRequest, ReferenceAudioInput},
    GeneratedAudioResult,
};
use serde_json::{json, Value};
use shine_rs::{encode_pcm_to_mp3, Mp3EncoderConfig, StereoMode};

use crate::{
    error::ServerError,
    openai::{SpeechRequest, SpeechResponseFormat, SpeechStreamFormat},
    runtime::AppState,
};

pub struct ParsedSpeechRequest {
    pub generation_request: GenerationRequest,
    pub response_format: SpeechResponseFormat,
    pub stream_format: SpeechStreamFormat,
    pub seed_override: Option<u64>,
}

struct SpeechRequestParts {
    model: String,
    input: String,
    voice: Option<Value>,
    instructions: Option<String>,
    response_format: String,
    speed: Option<f32>,
    stream_format: Option<String>,
    extra: HashMap<String, Value>,
    ref_audio_override: Option<ReferenceAudioInput>,
}

impl From<SpeechRequest> for SpeechRequestParts {
    fn from(value: SpeechRequest) -> Self {
        Self {
            model: value.model,
            input: value.input,
            voice: value.voice,
            instructions: value.instructions,
            response_format: value.response_format,
            speed: value.speed,
            stream_format: value.stream_format,
            extra: value.extra,
            ref_audio_override: None,
        }
    }
}

pub async fn parse_http_speech_request(
    request: Request,
    state: &AppState,
) -> Result<ParsedSpeechRequest, ServerError> {
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .to_ascii_lowercase()
                .starts_with("multipart/form-data")
        })
        .unwrap_or(false);

    let request = if is_multipart {
        parse_multipart_speech_request(request, state).await?
    } else {
        Json::<SpeechRequest>::from_request(request, state)
            .await
            .map(|Json(payload)| SpeechRequestParts::from(payload))
            .map_err(|error| ServerError::validation(error.to_string()))?
    };

    parse_speech_request(request, &state.config.served_model_id)
}

fn parse_speech_request(
    request: SpeechRequestParts,
    served_model_id: &str,
) -> Result<ParsedSpeechRequest, ServerError> {
    validate_model_name(&request.model, served_model_id)?;
    let _ = request.voice.as_ref();
    let response_format = parse_response_format(&request.response_format)?;
    let stream_format = parse_stream_format(request.stream_format.as_deref())?;
    let speed = request.speed.unwrap_or(1.0);
    if !speed.is_finite() || speed <= 0.0 {
        return Err(ServerError::validation(
            "speed must be finite and greater than zero",
        ));
    }

    let mut generation_config = GenerationConfig::default();
    set_generation_config_overrides(&request.extra, &mut generation_config)?;
    validate_generation_config(&generation_config)?;

    let mut generation_request = GenerationRequest::new_text_only(request.input);
    generation_request.generation_config = generation_config;
    generation_request.speeds = vec![Some(speed)];

    if let Some(language) = parse_optional_string(&request.extra, "language")? {
        generation_request.languages = vec![Some(language)];
    }
    if let Some(duration) = parse_optional_f32(&request.extra, "duration")? {
        if !duration.is_finite() || duration <= 0.0 {
            return Err(ServerError::validation(
                "duration must be finite and greater than zero",
            ));
        }
        generation_request.durations = vec![Some(duration)];
    }
    if let Some(asr_model) = parse_optional_string(&request.extra, "asr_model")? {
        generation_request.asr_model = Some(asr_model);
    }
    if let Some(ref_text) = parse_optional_string(&request.extra, "ref_text")? {
        generation_request.ref_texts = vec![Some(ref_text)];
    }

    let instruct = parse_optional_string(&request.extra, "instruct")?.or_else(|| {
        request
            .instructions
            .filter(|value| !value.trim().is_empty())
    });
    if let Some(instruct) = instruct {
        generation_request.instructs = vec![Some(instruct)];
    }

    if let Some(ref_audio) = request.ref_audio_override {
        generation_request.ref_audios = vec![Some(ref_audio)];
    } else if let Some(ref_audio) = request.extra.get("ref_audio") {
        generation_request.ref_audios = vec![Some(parse_data_uri_audio(ref_audio)?)];
    }
    let seed_override = parse_optional_u64(&request.extra, "seed")?;

    Ok(ParsedSpeechRequest {
        generation_request,
        response_format,
        stream_format,
        seed_override,
    })
}

async fn parse_multipart_speech_request(
    request: Request,
    state: &AppState,
) -> Result<SpeechRequestParts, ServerError> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|error| ServerError::validation(error.body_text()))?;

    let mut model = None;
    let mut input = None;
    let mut voice = None;
    let mut instructions = None;
    let mut response_format = "wav".to_string();
    let mut speed = None;
    let mut stream_format = None;
    let mut extra = HashMap::new();
    let mut ref_audio_override = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ServerError::validation(error.body_text()))?
    {
        let name = field
            .name()
            .map(str::to_string)
            .ok_or_else(|| ServerError::validation("multipart field name is missing"))?;

        match name.as_str() {
            "ref_audio" => {
                let file_name = field.file_name().map(str::to_string);
                let content_type = field.content_type().map(str::to_string);
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|error| ServerError::validation(error.body_text()))?;
                let extension_hint = file_name
                    .as_deref()
                    .and_then(extension_from_filename)
                    .or_else(|| {
                        content_type
                            .as_deref()
                            .and_then(|value| mime_to_extension(value))
                    });
                let waveform = load_audio_bytes(bytes.as_ref(), extension_hint)
                    .map_err(ServerError::from_infer)?;
                ref_audio_override = Some(ReferenceAudioInput::Waveform(waveform));
            }
            "model" => model = Some(read_multipart_text(field).await?),
            "input" => input = Some(read_multipart_text(field).await?),
            "voice" => {
                let raw = read_multipart_text(field).await?;
                let parsed = serde_json::from_str::<Value>(&raw).unwrap_or(Value::String(raw));
                voice = Some(parsed);
            }
            "instructions" => instructions = Some(read_multipart_text(field).await?),
            "response_format" => response_format = read_multipart_text(field).await?,
            "speed" => {
                speed = Some(parse_multipart_f32(
                    "speed",
                    &read_multipart_text(field).await?,
                )?)
            }
            "stream_format" => stream_format = Some(read_multipart_text(field).await?),
            other => {
                if field.file_name().is_some() {
                    return Err(ServerError::validation(format!(
                        "unsupported multipart file field `{other}`"
                    )));
                }
                let raw = read_multipart_text(field).await?;
                extra.insert(other.to_string(), parse_multipart_extra_value(other, &raw)?);
            }
        }
    }

    Ok(SpeechRequestParts {
        model: model.ok_or_else(|| ServerError::validation("model is required"))?,
        input: input.ok_or_else(|| ServerError::validation("input is required"))?,
        voice,
        instructions,
        response_format,
        speed,
        stream_format,
        extra,
        ref_audio_override,
    })
}

pub fn build_audio_response(
    result: GeneratedAudioResult,
    response_format: SpeechResponseFormat,
    stream_format: SpeechStreamFormat,
    mp3_bitrate_kbps: u32,
) -> Result<Response<Body>, ServerError> {
    let (content_type, payload) = encode_audio(&result.audio, response_format, mp3_bitrate_kbps)?;
    match stream_format {
        SpeechStreamFormat::Audio => Ok(binary_stream_response(content_type, payload)),
        SpeechStreamFormat::Sse => Ok(sse_stream_response(payload, result)),
    }
}

fn validate_model_name(model: &str, served_model_id: &str) -> Result<(), ServerError> {
    if model == "default" || model == served_model_id {
        return Ok(());
    }
    Err(ServerError::validation(format!(
        "model `{model}` is not available; expected `{served_model_id}` or `default`"
    )))
}

fn parse_response_format(value: &str) -> Result<SpeechResponseFormat, ServerError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "wav" => Ok(SpeechResponseFormat::Wav),
        "pcm" => Ok(SpeechResponseFormat::Pcm),
        "mp3" => Ok(SpeechResponseFormat::Mp3),
        other => Err(ServerError::validation(format!(
            "unsupported response_format `{other}`"
        ))),
    }
}

fn parse_stream_format(value: Option<&str>) -> Result<SpeechStreamFormat, ServerError> {
    match value.map(|item| item.trim().to_ascii_lowercase()) {
        None => Ok(SpeechStreamFormat::Audio),
        Some(value) if value == "audio" => Ok(SpeechStreamFormat::Audio),
        Some(value) if value == "sse" => Ok(SpeechStreamFormat::Sse),
        Some(other) => Err(ServerError::validation(format!(
            "unsupported stream_format `{other}`"
        ))),
    }
}

fn set_generation_config_overrides(
    values: &std::collections::HashMap<String, Value>,
    config: &mut GenerationConfig,
) -> Result<(), ServerError> {
    if let Some(value) = parse_optional_usize(values, "num_step")? {
        config.num_step = value;
    }
    if let Some(value) = parse_optional_f32(values, "guidance_scale")? {
        config.guidance_scale = value;
    }
    if let Some(value) = parse_optional_f32(values, "t_shift")? {
        config.t_shift = value;
    }
    if let Some(value) = parse_optional_f32(values, "layer_penalty_factor")? {
        config.layer_penalty_factor = value;
    }
    if let Some(value) = parse_optional_f32(values, "position_temperature")? {
        config.position_temperature = value;
    }
    if let Some(value) = parse_optional_f32(values, "class_temperature")? {
        config.class_temperature = value;
    }
    if let Some(value) = parse_optional_bool(values, "preprocess_prompt")? {
        config.preprocess_prompt = value;
    }
    if let Some(value) = parse_optional_bool(values, "postprocess_output")? {
        config.postprocess_output = value;
    }
    if let Some(value) = parse_optional_bool(values, "denoise")? {
        config.denoise = value;
    }
    if let Some(value) = parse_optional_f32(values, "audio_chunk_duration")? {
        config.audio_chunk_duration = value;
    }
    if let Some(value) = parse_optional_f32(values, "audio_chunk_threshold")? {
        config.audio_chunk_threshold = value;
    }
    Ok(())
}

fn validate_generation_config(config: &GenerationConfig) -> Result<(), ServerError> {
    const MAX_GENERATION_STEPS: usize = 4096;
    if config.num_step == 0 || config.num_step > MAX_GENERATION_STEPS {
        return Err(ServerError::validation(format!(
            "num_step must be in the range 1..={MAX_GENERATION_STEPS}"
        )));
    }
    if !config.guidance_scale.is_finite() {
        return Err(ServerError::validation("guidance_scale must be finite"));
    }
    if !config.t_shift.is_finite() || config.t_shift <= 0.0 {
        return Err(ServerError::validation(
            "t_shift must be finite and greater than zero",
        ));
    }
    if !config.layer_penalty_factor.is_finite() {
        return Err(ServerError::validation(
            "layer_penalty_factor must be finite",
        ));
    }
    if !config.position_temperature.is_finite() || config.position_temperature < 0.0 {
        return Err(ServerError::validation(
            "position_temperature must be finite and non-negative",
        ));
    }
    if !config.class_temperature.is_finite() || config.class_temperature < 0.0 {
        return Err(ServerError::validation(
            "class_temperature must be finite and non-negative",
        ));
    }
    if !config.audio_chunk_duration.is_finite() || config.audio_chunk_duration <= 0.0 {
        return Err(ServerError::validation(
            "audio_chunk_duration must be finite and greater than zero",
        ));
    }
    if !config.audio_chunk_threshold.is_finite() || config.audio_chunk_threshold <= 0.0 {
        return Err(ServerError::validation(
            "audio_chunk_threshold must be finite and greater than zero",
        ));
    }
    Ok(())
}

fn parse_data_uri_audio(value: &Value) -> Result<ReferenceAudioInput, ServerError> {
    let input = value
        .as_str()
        .ok_or_else(|| ServerError::validation("ref_audio must be a data URI string"))?;
    let Some(payload) = input.strip_prefix("data:") else {
        return Err(ServerError::validation("ref_audio must use a data URI"));
    };
    let Some((meta, encoded)) = payload.split_once(',') else {
        return Err(ServerError::validation("ref_audio data URI is malformed"));
    };
    let Some((mime, encoding)) = meta.rsplit_once(';') else {
        return Err(ServerError::validation(
            "ref_audio data URI must be base64-encoded",
        ));
    };
    if !encoding.eq_ignore_ascii_case("base64") {
        return Err(ServerError::validation(
            "ref_audio data URI must be base64-encoded",
        ));
    }

    let bytes = BASE64.decode(encoded).map_err(|error| {
        ServerError::validation(format!("invalid ref_audio base64 payload: {error}"))
    })?;
    let extension = mime_to_extension(mime);
    let waveform = load_audio_bytes(&bytes, extension).map_err(ServerError::from_infer)?;
    Ok(ReferenceAudioInput::Waveform(waveform))
}

fn mime_to_extension(mime: &str) -> Option<&'static str> {
    match mime.to_ascii_lowercase().as_str() {
        "audio/wav" | "audio/x-wav" | "audio/wave" => Some("wav"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/flac" => Some("flac"),
        "audio/ogg" => Some("ogg"),
        "audio/aac" => Some("aac"),
        "audio/mp4" | "audio/m4a" => Some("mp4"),
        _ => None,
    }
}

fn extension_from_filename(file_name: &str) -> Option<&str> {
    Path::new(file_name)
        .extension()
        .and_then(|value| value.to_str())
}

fn encode_audio(
    audio: &DecodedAudio,
    response_format: SpeechResponseFormat,
    mp3_bitrate_kbps: u32,
) -> Result<(&'static str, Vec<u8>), ServerError> {
    audio.validate().map_err(ServerError::from_infer)?;
    match response_format {
        SpeechResponseFormat::Wav => Ok(("audio/wav", encode_wav(audio)?)),
        SpeechResponseFormat::Pcm => Ok(("audio/pcm", encode_pcm(audio))),
        SpeechResponseFormat::Mp3 => Ok(("audio/mpeg", encode_mp3(audio, mp3_bitrate_kbps)?)),
    }
}

fn encode_wav(audio: &DecodedAudio) -> Result<Vec<u8>, ServerError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|error| ServerError::internal(error.to_string()))?;
        for sample in &audio.samples {
            writer
                .write_sample(*sample)
                .map_err(|error| ServerError::internal(error.to_string()))?;
        }
        writer
            .finalize()
            .map_err(|error| ServerError::internal(error.to_string()))?;
    }
    Ok(cursor.into_inner())
}

fn encode_pcm(audio: &DecodedAudio) -> Vec<u8> {
    float_samples_to_i16(audio)
        .into_iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

fn encode_mp3(audio: &DecodedAudio, bitrate_kbps: u32) -> Result<Vec<u8>, ServerError> {
    let config = Mp3EncoderConfig::new()
        .sample_rate(audio.sample_rate)
        .bitrate(bitrate_kbps)
        .channels(1)
        .stereo_mode(StereoMode::Mono);
    encode_pcm_to_mp3(config, &float_samples_to_i16(audio))
        .map_err(|error| ServerError::internal(error.to_string()))
}

fn float_samples_to_i16(audio: &DecodedAudio) -> Vec<i16> {
    audio
        .samples
        .iter()
        .map(|sample: &f32| (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)
        .collect()
}

fn binary_stream_response(content_type: &'static str, payload: Vec<u8>) -> Response<Body> {
    let chunks = payload
        .chunks(8192)
        .map(|chunk| Ok::<Bytes, Infallible>(Bytes::copy_from_slice(chunk)))
        .collect::<Vec<_>>();
    let body = Body::from_stream(stream::iter(chunks));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    response
}

fn sse_stream_response(payload: Vec<u8>, result: GeneratedAudioResult) -> Response<Body> {
    // Generation and encoding are complete before this endpoint is called; this is
    // post-generation chunking, but events are still produced lazily without a
    // second in-memory Vec of all SSE messages.
    let audio_events = stream::unfold((payload, 0usize), |(payload, offset)| async move {
        if offset >= payload.len() {
            return None;
        }
        let end = offset.saturating_add(4096).min(payload.len());
        let event = Ok::<Event, Infallible>(
            Event::default().data(
                json!({
                    "type": "speech.audio.delta",
                    "audio": BASE64.encode(&payload[offset..end]),
                })
                .to_string(),
            ),
        );
        Some((event, (payload, end)))
    });
    let terminal_events = stream::iter([
        Ok::<Event, Infallible>(
            Event::default().data(
                json!({
                    "type": "speech.audio.done",
                    "usage": {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                    }
                })
                .to_string(),
            ),
        ),
        Ok::<Event, Infallible>(Event::default().data("[DONE]")),
    ]);

    Sse::new(audio_events.chain(terminal_events))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn parse_optional_string(
    values: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<Option<String>, ServerError> {
    values
        .get(key)
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| ServerError::validation(format!("{key} must be a string")))
        })
        .transpose()
}

fn parse_optional_f32(
    values: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<Option<f32>, ServerError> {
    values
        .get(key)
        .map(|value| {
            let number = value
                .as_f64()
                .ok_or_else(|| ServerError::validation(format!("{key} must be a number")))?
                as f32;
            if !number.is_finite() {
                return Err(ServerError::validation(format!("{key} must be finite")));
            }
            Ok(number)
        })
        .transpose()
}

fn parse_optional_usize(
    values: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<Option<usize>, ServerError> {
    values
        .get(key)
        .map(|value| {
            let number = value
                .as_u64()
                .ok_or_else(|| ServerError::validation(format!("{key} must be an integer")))?;
            usize::try_from(number).map_err(|_| {
                ServerError::validation(format!("{key} is too large for this platform"))
            })
        })
        .transpose()
}

fn parse_optional_bool(
    values: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<Option<bool>, ServerError> {
    values
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| ServerError::validation(format!("{key} must be a boolean")))
        })
        .transpose()
}

fn parse_optional_u64(
    values: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<Option<u64>, ServerError> {
    values
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| ServerError::validation(format!("{key} must be an integer")))
        })
        .transpose()
}

async fn read_multipart_text(
    field: axum::extract::multipart::Field<'_>,
) -> Result<String, ServerError> {
    field
        .text()
        .await
        .map_err(|error| ServerError::validation(error.body_text()))
}

fn parse_multipart_extra_value(name: &str, raw: &str) -> Result<Value, ServerError> {
    match name {
        "duration"
        | "guidance_scale"
        | "t_shift"
        | "layer_penalty_factor"
        | "position_temperature"
        | "class_temperature"
        | "audio_chunk_duration"
        | "audio_chunk_threshold" => Ok(Value::from(parse_multipart_f64(name, raw)?)),
        "seed" | "num_step" => Ok(Value::from(parse_multipart_u64(name, raw)?)),
        "preprocess_prompt" | "postprocess_output" | "denoise" => {
            Ok(Value::from(parse_multipart_bool(name, raw)?))
        }
        _ => Ok(Value::String(raw.to_string())),
    }
}

fn parse_multipart_f32(name: &str, raw: &str) -> Result<f32, ServerError> {
    let value = raw
        .parse::<f32>()
        .map_err(|_| ServerError::validation(format!("{name} must be a number")))?;
    if !value.is_finite() {
        return Err(ServerError::validation(format!("{name} must be finite")));
    }
    Ok(value)
}

fn parse_multipart_f64(name: &str, raw: &str) -> Result<f64, ServerError> {
    let value = raw
        .parse::<f64>()
        .map_err(|_| ServerError::validation(format!("{name} must be a number")))?;
    if !value.is_finite() {
        return Err(ServerError::validation(format!("{name} must be finite")));
    }
    Ok(value)
}

fn parse_multipart_u64(name: &str, raw: &str) -> Result<u64, ServerError> {
    raw.parse::<u64>()
        .map_err(|_| ServerError::validation(format!("{name} must be an integer")))
}

fn parse_multipart_bool(name: &str, raw: &str) -> Result<bool, ServerError> {
    raw.parse::<bool>()
        .map_err(|_| ServerError::validation(format!("{name} must be a boolean")))
}
