use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, RwLock,
    },
    time::Duration,
};

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

    fn synthesize_with_seed(
        &self,
        request: GenerationRequest,
        seed: Option<u64>,
        cancellation: Arc<AtomicBool>,
    ) -> omnivoice_infer::Result<GeneratedAudioResult> {
        if cancellation.load(Ordering::Acquire) {
            return Err(omnivoice_infer::OmniVoiceError::InvalidRequest(
                "inference cancelled".to_string(),
            ));
        }
        if let Some(seed) = seed {
            self.set_seed(seed)?;
        }
        self.synthesize(request)
    }
}

pub struct PipelineSpeechRuntime {
    pipeline: Phase3Pipeline,
    inference_lock: Mutex<()>,
}

impl PipelineSpeechRuntime {
    pub fn from_options(options: RuntimeOptions) -> Result<Self, ServerError> {
        Ok(Self {
            pipeline: Phase3Pipeline::from_options(options)?,
            inference_lock: Mutex::new(()),
        })
    }
}

impl SpeechRuntime for PipelineSpeechRuntime {
    fn synthesize(
        &self,
        request: GenerationRequest,
    ) -> omnivoice_infer::Result<GeneratedAudioResult> {
        let _guard = self
            .inference_lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut results = self.pipeline.generate_with_usage(&request)?;
        results.pop().ok_or_else(|| {
            omnivoice_infer::OmniVoiceError::InvalidData(
                "pipeline did not return an audio result".to_string(),
            )
        })
    }

    fn set_seed(&self, seed: u64) -> omnivoice_infer::Result<()> {
        let _guard = self
            .inference_lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        self.pipeline.stage0().set_seed(seed)
    }

    fn synthesize_with_seed(
        &self,
        request: GenerationRequest,
        seed: Option<u64>,
        cancellation: Arc<AtomicBool>,
    ) -> omnivoice_infer::Result<GeneratedAudioResult> {
        let _guard = self
            .inference_lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        if cancellation.load(Ordering::Acquire) {
            return Err(omnivoice_infer::OmniVoiceError::InvalidRequest(
                "inference cancelled".to_string(),
            ));
        }
        self.pipeline
            .set_cancellation_flag(Some(Arc::clone(&cancellation)));

        let result = (|| {
            if let Some(seed) = seed {
                self.pipeline.stage0().set_seed(seed)?;
            }
            let mut results = self.pipeline.generate_with_usage(&request)?;
            results.pop().ok_or_else(|| {
                omnivoice_infer::OmniVoiceError::InvalidData(
                    "pipeline did not return an audio result".to_string(),
                )
            })
        })();

        self.pipeline.set_cancellation_flag(None);
        result
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub served_model_id: String,
    pub api_key: String,
    pub base_path: String,
    pub max_body_bytes: usize,
    pub max_concurrent_requests: usize,
    pub mp3_bitrate_kbps: u32,
    pub request_timeout: Duration,
}

impl ServerConfig {
    pub fn from_args(args: &ServerArgs) -> Result<Self, ServerError> {
        args.validate()?;
        Ok(Self {
            served_model_id: args.served_model_id.clone(),
            api_key: args.api_key.clone().unwrap_or_default(),
            base_path: args.normalized_base_path(),
            max_body_bytes: args.max_body_mb * 1024 * 1024,
            max_concurrent_requests: args.max_concurrent_requests,
            mp3_bitrate_kbps: args.mp3_bitrate_kbps,
            request_timeout: args.request_timeout(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeStatus {
    Starting,
    Ready,
    Failed,
}

#[derive(Clone)]
pub struct AppState {
    runtime: Arc<RwLock<Option<Arc<dyn SpeechRuntime>>>>,
    status: Arc<RwLock<RuntimeStatus>>,
    pub config: Arc<ServerConfig>,
    pub limiter: Arc<Semaphore>,
}

impl AppState {
    pub fn new<R>(runtime: R, config: ServerConfig) -> Self
    where
        R: SpeechRuntime + 'static,
    {
        let runtime: Arc<dyn SpeechRuntime> = Arc::new(runtime);
        let limiter = Arc::new(Semaphore::new(config.max_concurrent_requests));
        Self {
            runtime: Arc::new(RwLock::new(Some(runtime))),
            status: Arc::new(RwLock::new(RuntimeStatus::Ready)),
            config: Arc::new(config),
            limiter,
        }
    }

    pub fn starting(config: ServerConfig) -> Self {
        let limiter = Arc::new(Semaphore::new(config.max_concurrent_requests));
        Self {
            runtime: Arc::new(RwLock::new(None)),
            status: Arc::new(RwLock::new(RuntimeStatus::Starting)),
            config: Arc::new(config),
            limiter,
        }
    }

    pub fn install_runtime<R>(&self, runtime: R)
    where
        R: SpeechRuntime + 'static,
    {
        *self
            .runtime
            .write()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(Arc::new(runtime));
        *self
            .status
            .write()
            .unwrap_or_else(|poison| poison.into_inner()) = RuntimeStatus::Ready;
    }

    pub fn mark_failed(&self) {
        *self
            .status
            .write()
            .unwrap_or_else(|poison| poison.into_inner()) = RuntimeStatus::Failed;
    }

    pub fn runtime(&self) -> Option<Arc<dyn SpeechRuntime>> {
        self.runtime
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    pub fn status(&self) -> RuntimeStatus {
        *self
            .status
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}
