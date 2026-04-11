#[allow(dead_code)]
#[path = "../main.rs"]
mod cli_entrypoint;

use omnivoice_infer::OmniVoiceError;

fn main() -> Result<(), OmniVoiceError> {
    cli_entrypoint::run_from_env(cli_entrypoint::InvocationMode::InferWrapper)
}
