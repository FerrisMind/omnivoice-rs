#![deny(unsafe_op_in_unsafe_fn)]

pub mod artifacts;
pub mod asr;
pub mod audio_input;
pub mod audio_tokenizer;
pub mod contracts;
pub mod error;
pub mod frontend;
pub mod gpu_lock;
mod paths;
pub mod pipeline;
pub mod postprocess;
pub mod reference_prompt;
pub mod runtime;
pub mod stage0_loop;
pub mod stage0_model;
pub mod stage0_qwen3;
pub mod stage1_decoder;
pub mod stage1_model;

pub use contracts::{BoolTensor2, BoolTensor4, GeneratedAudioResult, GenerationUsage};
pub use error::{OmniVoiceError, Result};
pub use runtime::{DTypeSpec, DeviceSpec, RuntimeOptions};

pub fn workspace_phase_marker() -> &'static str {
    "omnivoice-phase10"
}

pub fn phase1_workspace_marker() -> &'static str {
    workspace_phase_marker()
}
