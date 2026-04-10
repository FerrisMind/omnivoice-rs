use std::{convert::Infallible, io::Cursor};

use axum::{
    body::{Body, Bytes},
    http::{header, HeaderValue, Response, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::stream;
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
};

pub struct ParsedSpeechRequest {
    pub generation_request: GenerationRequest,
    pub response_format: SpeechResponseFormat,
    pub stream_format: SpeechStreamFormat,
    pub seed_override: Option<u64>,
}

pub fn parse_speech_request(
    request: SpeechRequest,
    served_model_id: &str,
) -> Result<ParsedSpeechRequest, ServerError> {
    validate_model_name(&request.model, served_model_id)?;
    let _ = request.voice.as_ref();
    let response_format = parse_response_format(&request.response_format)?;
    let stream_format = parse_stream_format(request.stream_format.as_deref())?;
    let speed = request.speed.unwrap_or(1.0);
    if speed <= 0.0 {
        return Err(ServerError::validation("speed must be greater than zero"));
    }

    let mut generation_config = GenerationConfig::default();
    set_generation_config_overrides(&request.extra, &mut generation_config)?;

    let mut generation_request = GenerationRequest::new_text_only(request.input);
    generation_request.generation_config = generation_config;
    generation_request.speeds = vec![Some(speed)];

    if let Some(language) = parse_optional_string(&request.extra, "language")? {
        generation_request.languages = vec![Some(language)];
    }
    if let Some(duration) = parse_optional_f32(&request.extra, "duration")? {
        if duration <= 0.0 {
            return Err(ServerError::validation(
                "duration must be greater than zero",
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

    if let Some(ref_audio) = request.extra.get("ref_audio") {
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
    if !meta.ends_with(";base64") {
        return Err(ServerError::validation(
            "ref_audio data URI must be base64-encoded",
        ));
    }

    let mime = meta.trim_end_matches(";base64");
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

fn encode_audio(
    audio: &DecodedAudio,
    response_format: SpeechResponseFormat,
    mp3_bitrate_kbps: u32,
) -> Result<(&'static str, Vec<u8>), ServerError> {
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
    let mut events = payload
        .chunks(4096)
        .map(|chunk| {
            Ok::<Event, Infallible>(
                Event::default().data(
                    json!({
                        "type": "speech.audio.delta",
                        "audio": BASE64.encode(chunk),
                    })
                    .to_string(),
                ),
            )
        })
        .collect::<Vec<_>>();
    events.push(Ok(Event::default().data(
        json!({
            "type": "speech.audio.done",
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        })
        .to_string(),
    )));
    events.push(Ok(Event::default().data("[DONE]")));

    Sse::new(stream::iter(events))
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
            value
                .as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| ServerError::validation(format!("{key} must be a number")))
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
            value
                .as_u64()
                .map(|number| number as usize)
                .ok_or_else(|| ServerError::validation(format!("{key} must be an integer")))
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
