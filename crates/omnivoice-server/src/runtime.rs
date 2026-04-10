use std::sync::Arc;

use omnivoice_infer::{
    contracts::GenerationRequest, pipeline::Phase3Pipeline, GeneratedAudioResult, RuntimeOptions,
};
use tokio::sync::Semaphore;

use crate::{args::ServerArgs, error::ServerError};

pub trait SpeechRuntime: Send + Sync {
    fn synthesize(
        &self,
        request: GenerationRequest,
    ) -> omnivoice_infer::Result<GeneratedAudioResult>;

    fn set_seed(&self, _seed: u64) -> omnivoice_infer::Result<()> {
        Ok(())
    }
}

pub struct PipelineSpeechRuntime {
    pipeline: Phase3Pipeline,
}

impl PipelineSpeechRuntime {
    pub fn from_options(options: RuntimeOptions) -> Result<Self, ServerError> {
        Ok(Self {
            pipeline: Phase3Pipeline::from_options(options)?,
        })
    }
}

impl SpeechRuntime for PipelineSpeechRuntime {
    fn synthesize(
        &self,
        request: GenerationRequest,
    ) -> omnivoice_infer::Result<GeneratedAudioResult> {
        let mut results = self.pipeline.generate_with_usage(&request)?;
        results.pop().ok_or_else(|| {
            omnivoice_infer::OmniVoiceError::InvalidData(
                "pipeline did not return an audio result".to_string(),
            )
        })
    }

    fn set_seed(&self, seed: u64) -> omnivoice_infer::Result<()> {
        self.pipeline.stage0().set_seed(seed)
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub served_model_id: String,
    pub api_key: String,
    pub max_body_bytes: usize,
    pub max_concurrent_requests: usize,
    pub mp3_bitrate_kbps: u32,
}

impl ServerConfig {
    pub fn from_args(args: &ServerArgs) -> Result<Self, ServerError> {
        args.validate()?;
        Ok(Self {
            served_model_id: args.served_model_id.clone(),
            api_key: args.api_key.clone().unwrap_or_default(),
            max_body_bytes: args.max_body_mb * 1024 * 1024,
            max_concurrent_requests: args.max_concurrent_requests,
            mp3_bitrate_kbps: args.mp3_bitrate_kbps,
        })
    }
}

#[derive(Clone)]
pub struct AppState {
    pub runtime: Arc<dyn SpeechRuntime>,
    pub config: Arc<ServerConfig>,
    pub limiter: Arc<Semaphore>,
}

impl AppState {
    pub fn new<R>(runtime: R, config: ServerConfig) -> Self
    where
        R: SpeechRuntime + 'static,
    {
        let limiter = Arc::new(Semaphore::new(config.max_concurrent_requests));
        Self {
            runtime: Arc::new(runtime),
            config: Arc::new(config),
            limiter,
        }
    }
}
