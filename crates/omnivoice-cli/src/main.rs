use std::{
    collections::{BTreeSet, VecDeque},
    env,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};

use candle_core::Device;

use omnivoice_infer::{
    artifacts::{ReferenceArtifactBundle, RuntimeArtifacts},
    audio_input::ReferenceAudioProcessor,
    contracts::{GeneratedTokens, GenerationRequest, PreparedPromptSequence, ReferenceAudioInput},
    frontend::Frontend,
    model_source::{resolve_tts_model_root_from_path, DEFAULT_OMNIVOICE_REPO},
    pipeline::Phase3Pipeline,
    workspace_phase_marker, DTypeSpec, DeviceSpec, OmniVoiceError, RuntimeOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvocationMode {
    Multi,
    InferWrapper,
    InferBatchWrapper,
}

fn main() -> Result<(), OmniVoiceError> {
    run_from_env(InvocationMode::Multi)
}

pub fn run_from_env(mode: InvocationMode) -> Result<(), OmniVoiceError> {
    let args: Vec<String> = env::args().skip(1).collect();
    run_with_args(mode, &args)
}

fn run_with_args(mode: InvocationMode, args: &[String]) -> Result<(), OmniVoiceError> {
    if let Some(help) = requested_help(mode, args) {
        println!("{help}");
        return Ok(());
    }

    let command = parse_entrypoint_command(mode, args)?;
    execute_command(command)
}

fn parse_entrypoint_command(
    mode: InvocationMode,
    args: &[String],
) -> Result<CliCommand, OmniVoiceError> {
    let parsed = match mode {
        InvocationMode::Multi => CliCommand::parse(args),
        InvocationMode::InferWrapper => parse_wrapper_command("infer", args),
        InvocationMode::InferBatchWrapper => parse_wrapper_command("infer-batch", args),
    };

    parsed.map_err(|error| rewrite_usage_for_mode(mode, error))
}

fn parse_wrapper_command(command: &str, args: &[String]) -> Result<CliCommand, OmniVoiceError> {
    let mut forwarded = Vec::with_capacity(args.len() + 1);
    forwarded.push(command.to_string());
    forwarded.extend(args.iter().cloned());
    CliCommand::parse(&forwarded)
}

fn rewrite_usage_for_mode(mode: InvocationMode, error: OmniVoiceError) -> OmniVoiceError {
    let replacement = match mode {
        InvocationMode::Multi => return error,
        InvocationMode::InferWrapper => infer_usage("omnivoice-infer"),
        InvocationMode::InferBatchWrapper => infer_batch_usage("omnivoice-infer-batch"),
    };

    match error {
        OmniVoiceError::InvalidRequest(message) => {
            OmniVoiceError::InvalidRequest(message.replace(&usage(), &replacement))
        }
        other => other,
    }
}

fn requested_help(mode: InvocationMode, args: &[String]) -> Option<String> {
    match mode {
        InvocationMode::Multi => {
            if args.len() == 1 && is_help_flag(&args[0]) {
                return Some(usage());
            }

            let subcommand_help = args.iter().skip(1).any(|arg| is_help_flag(arg));
            match args.first().map(String::as_str) {
                Some("artifacts") if subcommand_help => {
                    Some(artifacts_validate_usage("omnivoice-cli artifacts"))
                }
                Some("infer") if subcommand_help => Some(infer_usage("omnivoice-cli infer")),
                Some("infer-batch") if subcommand_help => {
                    Some(infer_batch_usage("omnivoice-cli infer-batch"))
                }
                Some("prepare-prompt") if subcommand_help => {
                    Some(prepare_prompt_usage("omnivoice-cli prepare-prompt"))
                }
                Some("stage1-prepare") if subcommand_help => {
                    Some(stage1_prepare_usage("omnivoice-cli stage1-prepare"))
                }
                Some("stage1-decode") if subcommand_help => {
                    Some(stage1_decode_usage("omnivoice-cli stage1-decode"))
                }
                Some("stage0-generate") if subcommand_help => {
                    Some(stage0_generate_usage("omnivoice-cli stage0-generate"))
                }
                Some("stage0-debug") if subcommand_help => {
                    Some(stage0_debug_usage("omnivoice-cli stage0-debug"))
                }
                _ => None,
            }
        }
        InvocationMode::InferWrapper if args.iter().any(|arg| is_help_flag(arg)) => {
            Some(infer_usage("omnivoice-infer"))
        }
        InvocationMode::InferBatchWrapper if args.iter().any(|arg| is_help_flag(arg)) => {
            Some(infer_batch_usage("omnivoice-infer-batch"))
        }
        _ => None,
    }
}

fn is_help_flag(value: &str) -> bool {
    matches!(value, "--help" | "-h")
}

fn execute_command(command: CliCommand) -> Result<(), OmniVoiceError> {
    match command {
        CliCommand::ArtifactsValidate {
            model_dir,
            reference_root,
        } => run_artifacts_validate(model_dir, reference_root),
        CliCommand::Infer {
            model_dir,
            text,
            output,
            language,
            ref_audio,
            ref_text,
            instruct,
            duration,
            speed,
            asr_model,
            device,
            dtype,
            num_step,
            guidance_scale,
            t_shift,
            layer_penalty_factor,
            position_temperature,
            class_temperature,
            preprocess_prompt,
            postprocess_output,
            denoise,
            audio_chunk_duration,
            audio_chunk_threshold,
            seed,
        } => run_infer(
            model_dir,
            text,
            output,
            language,
            ref_audio,
            ref_text,
            instruct,
            duration,
            speed,
            asr_model,
            device,
            dtype,
            num_step,
            guidance_scale,
            t_shift,
            layer_penalty_factor,
            position_temperature,
            class_temperature,
            preprocess_prompt,
            postprocess_output,
            denoise,
            audio_chunk_duration,
            audio_chunk_threshold,
            seed,
        ),
        CliCommand::InferBatch {
            model_dir,
            test_list,
            res_dir,
            device,
            dtype,
            num_step,
            guidance_scale,
            t_shift,
            nj_per_gpu,
            audio_chunk_duration,
            audio_chunk_threshold,
            batch_duration,
            batch_size,
            warmup,
            preprocess_prompt,
            postprocess_output,
            layer_penalty_factor,
            position_temperature,
            class_temperature,
            denoise,
            lang_id,
            seed,
        } => run_infer_batch(
            model_dir,
            test_list,
            res_dir,
            device,
            dtype,
            num_step,
            guidance_scale,
            t_shift,
            nj_per_gpu,
            audio_chunk_duration,
            audio_chunk_threshold,
            batch_duration,
            batch_size,
            warmup,
            preprocess_prompt,
            postprocess_output,
            layer_penalty_factor,
            position_temperature,
            class_temperature,
            denoise,
            lang_id,
            seed,
        ),
        CliCommand::PreparePrompt {
            model_dir,
            reference_root,
            case,
            device,
            dtype,
        } => run_prepare_prompt(model_dir, reference_root, case, device, dtype),
        CliCommand::Stage1Prepare {
            model_dir,
            reference_root,
            case,
            device,
            dtype,
        } => run_stage1_prepare(model_dir, reference_root, case, device, dtype),
        CliCommand::Stage1Decode {
            model_dir,
            reference_root,
            case,
            out,
            raw,
            device,
            dtype,
        } => run_stage1_decode(model_dir, reference_root, case, out, raw, device, dtype),
        CliCommand::Stage0Generate {
            model_dir,
            reference_root,
            case,
            out,
            device,
            dtype,
        } => run_stage0_generate(model_dir, reference_root, case, out, device, dtype),
        CliCommand::Stage0Debug {
            model_dir,
            reference_root,
            case,
            device,
            dtype,
        } => run_stage0_debug(model_dir, reference_root, case, device, dtype),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum CliCommand {
    ArtifactsValidate {
        model_dir: PathBuf,
        reference_root: Option<PathBuf>,
    },
    Infer {
        model_dir: PathBuf,
        text: String,
        output: PathBuf,
        language: Option<String>,
        ref_audio: Option<PathBuf>,
        ref_text: Option<String>,
        instruct: Option<String>,
        duration: Option<f32>,
        speed: Option<f32>,
        asr_model: Option<String>,
        device: DeviceSpec,
        dtype: DTypeSpec,
        num_step: usize,
        guidance_scale: f32,
        t_shift: f32,
        layer_penalty_factor: f32,
        position_temperature: f32,
        class_temperature: f32,
        preprocess_prompt: bool,
        postprocess_output: bool,
        denoise: bool,
        audio_chunk_duration: f32,
        audio_chunk_threshold: f32,
        seed: Option<u64>,
    },
    InferBatch {
        model_dir: PathBuf,
        test_list: PathBuf,
        res_dir: PathBuf,
        device: DeviceSpec,
        dtype: DTypeSpec,
        num_step: usize,
        guidance_scale: f32,
        t_shift: f32,
        nj_per_gpu: usize,
        audio_chunk_duration: f32,
        audio_chunk_threshold: f32,
        batch_duration: f32,
        batch_size: usize,
        warmup: usize,
        preprocess_prompt: bool,
        postprocess_output: bool,
        layer_penalty_factor: f32,
        position_temperature: f32,
        class_temperature: f32,
        denoise: bool,
        lang_id: Option<String>,
        seed: Option<u64>,
    },
    PreparePrompt {
        model_dir: PathBuf,
        reference_root: PathBuf,
        case: String,
        device: DeviceSpec,
        dtype: DTypeSpec,
    },
    Stage1Prepare {
        model_dir: PathBuf,
        reference_root: PathBuf,
        case: String,
        device: DeviceSpec,
        dtype: DTypeSpec,
    },
    Stage1Decode {
        model_dir: PathBuf,
        reference_root: PathBuf,
        case: String,
        out: PathBuf,
        raw: bool,
        device: DeviceSpec,
        dtype: DTypeSpec,
    },
    Stage0Generate {
        model_dir: PathBuf,
        reference_root: PathBuf,
        case: String,
        out: PathBuf,
        device: DeviceSpec,
        dtype: DTypeSpec,
    },
    Stage0Debug {
        model_dir: PathBuf,
        reference_root: PathBuf,
        case: String,
        device: DeviceSpec,
        dtype: DTypeSpec,
    },
}

#[derive(Clone, Debug)]
struct BatchSample {
    id: String,
    text: String,
    ref_audio: Option<PathBuf>,
    ref_text: Option<String>,
    instruct: Option<String>,
    language: Option<String>,
    duration: Option<f32>,
    speed: Option<f32>,
}

#[derive(Clone, Debug)]
struct BatchJob {
    samples: Vec<BatchSample>,
}

#[derive(Clone, Debug, Default)]
struct BatchWorkerStats {
    batches_processed: usize,
    samples_written: usize,
}

impl CliCommand {
    fn parse(args: &[String]) -> Result<Self, OmniVoiceError> {
        match args.first().map(String::as_str) {
            Some("artifacts") => parse_artifacts_validate(args),
            Some("infer") => parse_infer(args),
            Some("infer-batch") => parse_infer_batch(args),
            Some("prepare-prompt") => parse_case_command(args, true),
            Some("stage1-prepare") => parse_case_command(args, false),
            Some("stage1-decode") => parse_stage1_decode(args),
            Some("stage0-generate") => parse_stage0_generate(args),
            Some("stage0-debug") => parse_stage0_debug(args),
            _ => Err(OmniVoiceError::InvalidRequest(usage())),
        }
    }
}

fn parse_artifacts_validate(args: &[String]) -> Result<CliCommand, OmniVoiceError> {
    if args.len() < 2 || args[1] != "validate" {
        return Err(OmniVoiceError::InvalidRequest(usage()));
    }

    let mut model_dir = None;
    let mut reference_root = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--model-dir" | "--model" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--model requires a path or repo id".to_string())
                })?;
                model_dir = Some(PathBuf::from(value));
                index += 2;
            }
            "--reference-root" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest(
                        "--reference-root requires a path value".to_string(),
                    )
                })?;
                reference_root = Some(PathBuf::from(value));
                index += 2;
            }
            other => {
                return Err(OmniVoiceError::InvalidRequest(format!(
                    "unknown argument: {other}\n{}",
                    usage()
                )));
            }
        }
    }

    Ok(CliCommand::ArtifactsValidate {
        model_dir: model_arg_or_default(model_dir),
        reference_root,
    })
}

fn parse_infer(args: &[String]) -> Result<CliCommand, OmniVoiceError> {
    let mut model_dir = None;
    let mut text = None;
    let mut output = None;
    let mut language = None;
    let mut ref_audio = None;
    let mut ref_text = None;
    let mut instruct = None;
    let mut duration = None;
    let mut speed = None;
    let mut asr_model = None;
    let mut device = DeviceSpec::default();
    let mut dtype = DTypeSpec::default();
    let mut num_step = 32usize;
    let mut guidance_scale = 2.0f32;
    let mut t_shift = 0.1f32;
    let mut layer_penalty_factor = 5.0f32;
    let mut position_temperature = 5.0f32;
    let mut class_temperature = 0.0f32;
    let mut preprocess_prompt = true;
    let mut postprocess_output = true;
    let mut denoise = true;
    let mut audio_chunk_duration = 15.0f32;
    let mut audio_chunk_threshold = 30.0f32;
    let mut seed = None;
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--model-dir" | "--model" => {
                model_dir = Some(PathBuf::from(required_value(args, index, "--model")?));
                index += 2;
            }
            "--text" => {
                text = Some(required_value(args, index, "--text")?.to_string());
                index += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(required_value(args, index, "--output")?));
                index += 2;
            }
            "--language" => {
                language = Some(required_value(args, index, "--language")?.to_string());
                index += 2;
            }
            "--ref-audio" | "--ref_audio" => {
                ref_audio = Some(PathBuf::from(required_value(args, index, "--ref-audio")?));
                index += 2;
            }
            "--ref-text" | "--ref_text" => {
                ref_text = Some(required_value(args, index, "--ref-text")?.to_string());
                index += 2;
            }
            "--instruct" => {
                instruct = Some(required_value(args, index, "--instruct")?.to_string());
                index += 2;
            }
            "--duration" => {
                duration = Some(parse_f32_arg(args, index, "--duration")?);
                index += 2;
            }
            "--speed" => {
                speed = Some(parse_f32_arg(args, index, "--speed")?);
                index += 2;
            }
            "--asr-model" => {
                asr_model = Some(required_value(args, index, "--asr-model")?.to_string());
                index += 2;
            }
            "--device" => {
                device = DeviceSpec::parse(required_value(args, index, "--device")?)?;
                index += 2;
            }
            "--dtype" => {
                dtype = DTypeSpec::parse(required_value(args, index, "--dtype")?)?;
                index += 2;
            }
            "--num-step" | "--num_step" => {
                num_step = parse_usize_arg(args, index, "--num-step")?;
                index += 2;
            }
            "--guidance-scale" | "--guidance_scale" => {
                guidance_scale = parse_f32_arg(args, index, "--guidance-scale")?;
                index += 2;
            }
            "--t-shift" | "--t_shift" => {
                t_shift = parse_f32_arg(args, index, "--t-shift")?;
                index += 2;
            }
            "--layer-penalty-factor" | "--layer_penalty_factor" => {
                layer_penalty_factor = parse_f32_arg(args, index, "--layer-penalty-factor")?;
                index += 2;
            }
            "--position-temperature" | "--position_temperature" => {
                position_temperature = parse_f32_arg(args, index, "--position-temperature")?;
                index += 2;
            }
            "--class-temperature" | "--class_temperature" => {
                class_temperature = parse_f32_arg(args, index, "--class-temperature")?;
                index += 2;
            }
            "--preprocess-prompt" | "--preprocess_prompt" => {
                preprocess_prompt = parse_bool_arg(args, index, "--preprocess-prompt")?;
                index += 2;
            }
            "--postprocess-output" | "--postprocess_output" => {
                postprocess_output = parse_bool_arg(args, index, "--postprocess-output")?;
                index += 2;
            }
            "--denoise" => {
                denoise = parse_bool_arg(args, index, "--denoise")?;
                index += 2;
            }
            "--audio-chunk-duration" | "--audio_chunk_duration" => {
                audio_chunk_duration = parse_f32_arg(args, index, "--audio-chunk-duration")?;
                index += 2;
            }
            "--audio-chunk-threshold" | "--audio_chunk_threshold" => {
                audio_chunk_threshold = parse_f32_arg(args, index, "--audio-chunk-threshold")?;
                index += 2;
            }
            "--seed" => {
                seed = Some(parse_u64_arg(args, index, "--seed")?);
                index += 2;
            }
            other => {
                return Err(OmniVoiceError::InvalidRequest(format!(
                    "unknown argument: {other}\n{}",
                    usage()
                )));
            }
        }
    }

    Ok(CliCommand::Infer {
        model_dir: model_arg_or_default(model_dir),
        text: required_string_arg(text, "--text")?,
        output: required_path_arg(output, "--output")?,
        language,
        ref_audio,
        ref_text,
        instruct,
        duration,
        speed,
        asr_model,
        device,
        dtype,
        num_step,
        guidance_scale,
        t_shift,
        layer_penalty_factor,
        position_temperature,
        class_temperature,
        preprocess_prompt,
        postprocess_output,
        denoise,
        audio_chunk_duration,
        audio_chunk_threshold,
        seed,
    })
}

fn parse_infer_batch(args: &[String]) -> Result<CliCommand, OmniVoiceError> {
    let mut model_dir = None;
    let mut test_list = None;
    let mut res_dir = None;
    let mut device = DeviceSpec::default();
    let mut dtype = DTypeSpec::default();
    let mut num_step = 32usize;
    let mut guidance_scale = 2.0f32;
    let mut t_shift = 0.1f32;
    let mut nj_per_gpu = 1usize;
    let mut audio_chunk_duration = 15.0f32;
    let mut audio_chunk_threshold = 30.0f32;
    let mut batch_duration = 1000.0f32;
    let mut batch_size = 0usize;
    let mut warmup = 0usize;
    let mut preprocess_prompt = true;
    let mut postprocess_output = true;
    let mut layer_penalty_factor = 5.0f32;
    let mut position_temperature = 5.0f32;
    let mut class_temperature = 0.0f32;
    let mut denoise = true;
    let mut lang_id = None;
    let mut seed = None;
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--model-dir" | "--model" => {
                model_dir = Some(PathBuf::from(required_value(args, index, "--model")?));
                index += 2;
            }
            "--test-list" | "--test_list" => {
                test_list = Some(PathBuf::from(required_value(args, index, "--test-list")?));
                index += 2;
            }
            "--res-dir" | "--res_dir" => {
                res_dir = Some(PathBuf::from(required_value(args, index, "--res-dir")?));
                index += 2;
            }
            "--device" => {
                device = DeviceSpec::parse(required_value(args, index, "--device")?)?;
                index += 2;
            }
            "--dtype" => {
                dtype = DTypeSpec::parse(required_value(args, index, "--dtype")?)?;
                index += 2;
            }
            "--num-step" | "--num_step" => {
                num_step = parse_usize_arg(args, index, "--num-step")?;
                index += 2;
            }
            "--guidance-scale" | "--guidance_scale" => {
                guidance_scale = parse_f32_arg(args, index, "--guidance-scale")?;
                index += 2;
            }
            "--t-shift" | "--t_shift" => {
                t_shift = parse_f32_arg(args, index, "--t-shift")?;
                index += 2;
            }
            "--nj-per-gpu" | "--nj_per_gpu" => {
                nj_per_gpu = parse_usize_arg(args, index, "--nj-per-gpu")?;
                index += 2;
            }
            "--audio-chunk-duration" | "--audio_chunk_duration" => {
                audio_chunk_duration = parse_f32_arg(args, index, "--audio-chunk-duration")?;
                index += 2;
            }
            "--audio-chunk-threshold" | "--audio_chunk_threshold" => {
                audio_chunk_threshold = parse_f32_arg(args, index, "--audio-chunk-threshold")?;
                index += 2;
            }
            "--batch-duration" | "--batch_duration" => {
                batch_duration = parse_f32_arg(args, index, "--batch-duration")?;
                index += 2;
            }
            "--batch-size" | "--batch_size" => {
                batch_size = parse_usize_arg(args, index, "--batch-size")?;
                index += 2;
            }
            "--warmup" => {
                warmup = parse_usize_arg(args, index, "--warmup")?;
                index += 2;
            }
            "--preprocess-prompt" | "--preprocess_prompt" => {
                preprocess_prompt = parse_bool_arg(args, index, "--preprocess-prompt")?;
                index += 2;
            }
            "--postprocess-output" | "--postprocess_output" => {
                postprocess_output = parse_bool_arg(args, index, "--postprocess-output")?;
                index += 2;
            }
            "--layer-penalty-factor" | "--layer_penalty_factor" => {
                layer_penalty_factor = parse_f32_arg(args, index, "--layer-penalty-factor")?;
                index += 2;
            }
            "--position-temperature" | "--position_temperature" => {
                position_temperature = parse_f32_arg(args, index, "--position-temperature")?;
                index += 2;
            }
            "--class-temperature" | "--class_temperature" => {
                class_temperature = parse_f32_arg(args, index, "--class-temperature")?;
                index += 2;
            }
            "--denoise" => {
                denoise = parse_bool_arg(args, index, "--denoise")?;
                index += 2;
            }
            "--lang-id" | "--lang_id" => {
                lang_id = Some(required_value(args, index, "--lang-id")?.to_string());
                index += 2;
            }
            "--seed" => {
                seed = Some(parse_u64_arg(args, index, "--seed")?);
                index += 2;
            }
            other => {
                return Err(OmniVoiceError::InvalidRequest(format!(
                    "unknown argument: {other}\n{}",
                    usage()
                )));
            }
        }
    }

    Ok(CliCommand::InferBatch {
        model_dir: model_arg_or_default(model_dir),
        test_list: required_path_arg(test_list, "--test-list")?,
        res_dir: required_path_arg(res_dir, "--res-dir")?,
        device,
        dtype,
        num_step,
        guidance_scale,
        t_shift,
        nj_per_gpu,
        audio_chunk_duration,
        audio_chunk_threshold,
        batch_duration,
        batch_size,
        warmup,
        preprocess_prompt,
        postprocess_output,
        layer_penalty_factor,
        position_temperature,
        class_temperature,
        denoise,
        lang_id,
        seed,
    })
}

fn parse_case_command(args: &[String], prompt: bool) -> Result<CliCommand, OmniVoiceError> {
    let mut model_dir = None;
    let mut reference_root = None;
    let mut case = None;
    let mut device = DeviceSpec::default();
    let mut dtype = DTypeSpec::default();
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--model-dir" | "--model" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--model requires a path or repo id".to_string())
                })?;
                model_dir = Some(PathBuf::from(value));
                index += 2;
            }
            "--reference-root" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest(
                        "--reference-root requires a path value".to_string(),
                    )
                })?;
                reference_root = Some(PathBuf::from(value));
                index += 2;
            }
            "--case" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--case requires an id value".to_string())
                })?;
                case = Some(value.clone());
                index += 2;
            }
            "--device" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--device requires a value".to_string())
                })?;
                device = DeviceSpec::parse(value)?;
                index += 2;
            }
            "--dtype" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--dtype requires a value".to_string())
                })?;
                dtype = DTypeSpec::parse(value)?;
                index += 2;
            }
            other => {
                return Err(OmniVoiceError::InvalidRequest(format!(
                    "unknown argument: {other}\n{}",
                    usage()
                )));
            }
        }
    }

    let Some(reference_root) = reference_root else {
        return Err(OmniVoiceError::InvalidRequest(format!(
            "missing required --reference-root\n{}",
            usage()
        )));
    };
    let Some(case) = case else {
        return Err(OmniVoiceError::InvalidRequest(format!(
            "missing required --case\n{}",
            usage()
        )));
    };

    if prompt {
        Ok(CliCommand::PreparePrompt {
            model_dir: model_arg_or_default(model_dir),
            reference_root,
            case,
            device,
            dtype,
        })
    } else {
        Ok(CliCommand::Stage1Prepare {
            model_dir: model_arg_or_default(model_dir),
            reference_root,
            case,
            device,
            dtype,
        })
    }
}

fn parse_stage1_decode(args: &[String]) -> Result<CliCommand, OmniVoiceError> {
    let mut model_dir = None;
    let mut reference_root = None;
    let mut case = None;
    let mut out = None;
    let mut raw = false;
    let mut device = DeviceSpec::default();
    let mut dtype = DTypeSpec::default();
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--model-dir" | "--model" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--model requires a path or repo id".to_string())
                })?;
                model_dir = Some(PathBuf::from(value));
                index += 2;
            }
            "--reference-root" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest(
                        "--reference-root requires a path value".to_string(),
                    )
                })?;
                reference_root = Some(PathBuf::from(value));
                index += 2;
            }
            "--case" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--case requires an id value".to_string())
                })?;
                case = Some(value.clone());
                index += 2;
            }
            "--out" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--out requires a path value".to_string())
                })?;
                out = Some(PathBuf::from(value));
                index += 2;
            }
            "--raw" => {
                raw = true;
                index += 1;
            }
            "--device" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--device requires a value".to_string())
                })?;
                device = DeviceSpec::parse(value)?;
                index += 2;
            }
            "--dtype" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--dtype requires a value".to_string())
                })?;
                dtype = DTypeSpec::parse(value)?;
                index += 2;
            }
            other => {
                return Err(OmniVoiceError::InvalidRequest(format!(
                    "unknown argument: {other}\n{}",
                    usage()
                )));
            }
        }
    }

    let Some(reference_root) = reference_root else {
        return Err(OmniVoiceError::InvalidRequest(format!(
            "missing required --reference-root\n{}",
            usage()
        )));
    };
    let Some(case) = case else {
        return Err(OmniVoiceError::InvalidRequest(format!(
            "missing required --case\n{}",
            usage()
        )));
    };
    let Some(out) = out else {
        return Err(OmniVoiceError::InvalidRequest(format!(
            "missing required --out\n{}",
            usage()
        )));
    };

    Ok(CliCommand::Stage1Decode {
        model_dir: model_arg_or_default(model_dir),
        reference_root,
        case,
        out,
        raw,
        device,
        dtype,
    })
}

fn parse_stage0_generate(args: &[String]) -> Result<CliCommand, OmniVoiceError> {
    let mut model_dir = None;
    let mut reference_root = None;
    let mut case = None;
    let mut out = None;
    let mut device = DeviceSpec::default();
    let mut dtype = DTypeSpec::default();
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--model-dir" | "--model" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--model requires a path or repo id".to_string())
                })?;
                model_dir = Some(PathBuf::from(value));
                index += 2;
            }
            "--reference-root" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest(
                        "--reference-root requires a path value".to_string(),
                    )
                })?;
                reference_root = Some(PathBuf::from(value));
                index += 2;
            }
            "--case" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--case requires an id value".to_string())
                })?;
                case = Some(value.clone());
                index += 2;
            }
            "--out" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--out requires a path value".to_string())
                })?;
                out = Some(PathBuf::from(value));
                index += 2;
            }
            "--device" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--device requires a value".to_string())
                })?;
                device = DeviceSpec::parse(value)?;
                index += 2;
            }
            "--dtype" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--dtype requires a value".to_string())
                })?;
                dtype = DTypeSpec::parse(value)?;
                index += 2;
            }
            other => {
                return Err(OmniVoiceError::InvalidRequest(format!(
                    "unknown argument: {other}\n{}",
                    usage()
                )));
            }
        }
    }

    Ok(CliCommand::Stage0Generate {
        model_dir: model_arg_or_default(model_dir),
        reference_root: required_path_arg(reference_root, "--reference-root")?,
        case: required_string_arg(case, "--case")?,
        out: required_path_arg(out, "--out")?,
        device,
        dtype,
    })
}

fn parse_stage0_debug(args: &[String]) -> Result<CliCommand, OmniVoiceError> {
    let mut model_dir = None;
    let mut reference_root = None;
    let mut case = None;
    let mut device = DeviceSpec::default();
    let mut dtype = DTypeSpec::default();
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--model-dir" | "--model" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--model requires a path or repo id".to_string())
                })?;
                model_dir = Some(PathBuf::from(value));
                index += 2;
            }
            "--reference-root" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest(
                        "--reference-root requires a path value".to_string(),
                    )
                })?;
                reference_root = Some(PathBuf::from(value));
                index += 2;
            }
            "--case" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--case requires an id value".to_string())
                })?;
                case = Some(value.clone());
                index += 2;
            }
            "--device" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--device requires a value".to_string())
                })?;
                device = DeviceSpec::parse(value)?;
                index += 2;
            }
            "--dtype" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    OmniVoiceError::InvalidRequest("--dtype requires a value".to_string())
                })?;
                dtype = DTypeSpec::parse(value)?;
                index += 2;
            }
            other => {
                return Err(OmniVoiceError::InvalidRequest(format!(
                    "unknown argument: {other}\n{}",
                    usage()
                )));
            }
        }
    }

    Ok(CliCommand::Stage0Debug {
        model_dir: model_arg_or_default(model_dir),
        reference_root: required_path_arg(reference_root, "--reference-root")?,
        case: required_string_arg(case, "--case")?,
        device,
        dtype,
    })
}

fn resolve_model_root(model: &Path) -> Result<PathBuf, OmniVoiceError> {
    resolve_tts_model_root_from_path(Some(model))
}

fn run_artifacts_validate(
    model_dir: PathBuf,
    reference_root: Option<PathBuf>,
) -> Result<(), OmniVoiceError> {
    let model_dir = resolve_model_root(&model_dir)?;
    let runtime = RuntimeArtifacts::from_model_root(&model_dir)?;
    println!("phase_marker={}", workspace_phase_marker());
    println!("runtime_manifest={}", runtime.manifest_path().display());
    println!("model_root={}", runtime.model_root().display());
    println!(
        "generator_config={}",
        runtime.generator().config_path().display()
    );
    println!(
        "generator_weights={}",
        runtime.generator().weights_path().display()
    );
    println!(
        "generator_prefixes={}",
        join_paths(runtime.generator().observed_prefixes())
    );
    println!(
        "text_tokenizer={}",
        runtime.text_tokenizer().tokenizer_path().display()
    );
    println!(
        "text_tokenizer_config={}",
        runtime.text_tokenizer().tokenizer_config_path().display()
    );
    println!(
        "chat_template={}",
        runtime
            .text_tokenizer()
            .chat_template_path()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<missing-optional>".to_string())
    );
    println!(
        "audio_tokenizer_config={}",
        runtime.audio_tokenizer().config_path().display()
    );
    println!(
        "audio_tokenizer_weights={}",
        runtime.audio_tokenizer().weights_path().display()
    );
    println!(
        "audio_tokenizer_preprocessor={}",
        runtime
            .audio_tokenizer()
            .preprocessor_config_path()
            .display()
    );
    println!(
        "audio_tokenizer_prefixes={}",
        join_paths(runtime.audio_tokenizer().observed_prefixes())
    );
    println!(
        "contracts=codebooks:{} vocab:{} mask:{} token_range:{}..={} sample_rate:{} hop_length:{} frame_rate:{}",
        runtime.contracts().num_audio_codebooks,
        runtime.contracts().audio_vocab_size,
        runtime.contracts().audio_mask_id,
        runtime.contracts().token_id_min,
        runtime.contracts().token_id_max,
        runtime.contracts().sample_rate,
        runtime.contracts().hop_length,
        runtime.contracts().frame_rate,
    );

    if let Some(reference_root) = reference_root {
        let reference = ReferenceArtifactBundle::from_root(reference_root)?;
        let case_ids = reference.available_case_ids()?;
        println!("reference_cases={}", case_ids.join(","));
        println!("reference_case_count={}", case_ids.len());
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_infer(
    model_dir: PathBuf,
    text: String,
    output: PathBuf,
    language: Option<String>,
    ref_audio: Option<PathBuf>,
    ref_text: Option<String>,
    instruct: Option<String>,
    duration: Option<f32>,
    speed: Option<f32>,
    asr_model: Option<String>,
    device: DeviceSpec,
    dtype: DTypeSpec,
    num_step: usize,
    guidance_scale: f32,
    t_shift: f32,
    layer_penalty_factor: f32,
    position_temperature: f32,
    class_temperature: f32,
    preprocess_prompt: bool,
    postprocess_output: bool,
    denoise: bool,
    audio_chunk_duration: f32,
    audio_chunk_threshold: f32,
    seed: Option<u64>,
) -> Result<(), OmniVoiceError> {
    let model_dir = resolve_model_root(&model_dir)?;
    let mut options = RuntimeOptions::new(model_dir)
        .with_device(device)
        .with_dtype(dtype);
    if let Some(seed) = seed {
        options = options.with_seed(seed);
    }
    let pipeline = Phase3Pipeline::from_options(options.clone())?;
    let mut request = GenerationRequest::new_text_only(text);
    request.languages = vec![language];
    request.ref_texts = vec![ref_text];
    request.instructs = vec![instruct];
    request.ref_audios =
        vec![ref_audio.map(|path| ReferenceAudioInput::from_path(path.display().to_string()))];
    request.speeds = vec![speed];
    request.durations = vec![duration];
    request.asr_model = asr_model;
    request.generation_config.num_step = num_step;
    request.generation_config.guidance_scale = guidance_scale;
    request.generation_config.t_shift = t_shift;
    request.generation_config.layer_penalty_factor = layer_penalty_factor;
    request.generation_config.position_temperature = position_temperature;
    request.generation_config.class_temperature = class_temperature;
    request.generation_config.preprocess_prompt = preprocess_prompt;
    request.generation_config.postprocess_output = postprocess_output;
    request.generation_config.denoise = denoise;
    request.generation_config.audio_chunk_duration = audio_chunk_duration;
    request.generation_config.audio_chunk_threshold = audio_chunk_threshold;

    let generated = pipeline.generate(&request)?;
    let audio = generated.into_iter().next().ok_or_else(|| {
        OmniVoiceError::InvalidData("live inference did not generate any audio".to_string())
    })?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    audio.write_wav(&output)?;

    println!("phase_marker={}", workspace_phase_marker());
    println!("command=infer");
    println!("model_root={}", options.model_root().display());
    println!("device={:?}", options.device());
    println!("dtype={:?}", options.dtype());
    println!(
        "resolved_device={}",
        describe_runtime_device(pipeline.stage0().device())
    );
    println!("resolved_dtype={:?}", pipeline.stage0().runtime_dtype());
    println!(
        "seed={}",
        options
            .seed()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "<none>".to_string())
    );
    println!("output={}", output.display());
    println!("sample_rate={}", audio.sample_rate);
    println!("frame_count={}", audio.frame_count());

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_infer_batch(
    model_dir: PathBuf,
    test_list: PathBuf,
    res_dir: PathBuf,
    device: DeviceSpec,
    dtype: DTypeSpec,
    num_step: usize,
    guidance_scale: f32,
    t_shift: f32,
    nj_per_gpu: usize,
    audio_chunk_duration: f32,
    audio_chunk_threshold: f32,
    batch_duration: f32,
    batch_size: usize,
    warmup: usize,
    preprocess_prompt: bool,
    postprocess_output: bool,
    layer_penalty_factor: f32,
    position_temperature: f32,
    class_temperature: f32,
    denoise: bool,
    lang_id: Option<String>,
    seed: Option<u64>,
) -> Result<(), OmniVoiceError> {
    if nj_per_gpu == 0 {
        return Err(OmniVoiceError::InvalidRequest(
            "--nj-per-gpu must be > 0".to_string(),
        ));
    }
    if batch_duration <= 0.0 && batch_size == 0 {
        return Err(OmniVoiceError::InvalidRequest(
            "--batch-duration must be > 0 when --batch-size is 0".to_string(),
        ));
    }

    let model_dir = resolve_model_root(&model_dir)?;
    fs::create_dir_all(&res_dir)?;
    let samples = read_test_list(&test_list, lang_id.as_deref())?;
    if samples.is_empty() {
        return Err(OmniVoiceError::InvalidRequest(format!(
            "test list {} does not contain any valid samples",
            test_list.display()
        )));
    }

    let frontend = Frontend::from_model_root(&model_dir)?;
    let processor = ReferenceAudioProcessor::new(24_000, 960);
    let jobs = if batch_size > 0 {
        cluster_samples_by_batch_size(&samples, &frontend, &processor, batch_size)?
    } else {
        cluster_samples_by_duration(&samples, &frontend, &processor, batch_duration)?
    };
    let worker_devices = resolve_batch_devices(device, nj_per_gpu)?;
    let queue = Arc::new(Mutex::new(VecDeque::from(jobs.clone())));
    let model_dir = Arc::new(model_dir);
    let res_dir = Arc::new(res_dir);

    let mut handles = Vec::with_capacity(worker_devices.len());
    for worker_device in worker_devices.clone() {
        let queue = Arc::clone(&queue);
        let model_dir = Arc::clone(&model_dir);
        let res_dir = Arc::clone(&res_dir);
        handles.push(thread::spawn(move || {
            run_batch_worker(
                worker_device,
                queue,
                model_dir,
                res_dir,
                dtype,
                num_step,
                guidance_scale,
                t_shift,
                audio_chunk_duration,
                audio_chunk_threshold,
                warmup,
                preprocess_prompt,
                postprocess_output,
                layer_penalty_factor,
                position_temperature,
                class_temperature,
                denoise,
                seed,
            )
        }));
    }

    let mut totals = BatchWorkerStats::default();
    for handle in handles {
        let stats = handle.join().map_err(|_| {
            OmniVoiceError::InvalidData("batch worker thread panicked".to_string())
        })??;
        totals.batches_processed += stats.batches_processed;
        totals.samples_written += stats.samples_written;
    }

    println!("phase_marker={}", workspace_phase_marker());
    println!("command=infer-batch");
    println!("model_root={}", model_dir.display());
    println!("test_list={}", test_list.display());
    println!("res_dir={}", res_dir.display());
    println!("device={:?}", device);
    println!("dtype={:?}", dtype);
    println!("worker_count={}", worker_devices.len());
    println!("resolved_workers={}", worker_devices.len());
    println!(
        "resolved_worker_devices={}",
        worker_devices
            .iter()
            .map(|device| format!("{device:?}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "resolved_worker_dtypes={}",
        worker_devices
            .iter()
            .map(|device| format!("{:?}", dtype.resolve_for_device(*device)))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("batch_count={}", jobs.len());
    println!("sample_count={}", samples.len());
    println!("written_files={}", totals.samples_written);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_batch_worker(
    worker_device: DeviceSpec,
    queue: Arc<Mutex<VecDeque<BatchJob>>>,
    model_dir: Arc<PathBuf>,
    res_dir: Arc<PathBuf>,
    dtype: DTypeSpec,
    num_step: usize,
    guidance_scale: f32,
    t_shift: f32,
    audio_chunk_duration: f32,
    audio_chunk_threshold: f32,
    warmup: usize,
    preprocess_prompt: bool,
    postprocess_output: bool,
    layer_penalty_factor: f32,
    position_temperature: f32,
    class_temperature: f32,
    denoise: bool,
    seed: Option<u64>,
) -> Result<BatchWorkerStats, OmniVoiceError> {
    let mut options = RuntimeOptions::new((*model_dir).clone())
        .with_device(worker_device)
        .with_dtype(dtype);
    if let Some(seed) = seed {
        options = options.with_seed(seed);
    }
    let pipeline = Phase3Pipeline::from_options(options)?;
    for _ in 0..warmup {
        let mut warmup_request = GenerationRequest::new_text_only("hello");
        warmup_request.languages = vec![Some("en".to_string())];
        let _ = pipeline.generate(&warmup_request)?;
    }

    let mut stats = BatchWorkerStats::default();
    loop {
        let job = {
            let mut guard = queue.lock().unwrap_or_else(|poison| poison.into_inner());
            guard.pop_front()
        };
        let Some(job) = job else { break };
        let request = build_batch_request(
            &job.samples,
            num_step,
            guidance_scale,
            t_shift,
            audio_chunk_duration,
            audio_chunk_threshold,
            preprocess_prompt,
            postprocess_output,
            layer_penalty_factor,
            position_temperature,
            class_temperature,
            denoise,
        );
        let audios = pipeline.generate(&request)?;
        for (sample, audio) in job.samples.iter().zip(audios.into_iter()) {
            let output_path = res_dir.join(format!("{}.wav", sample.id));
            audio.write_wav(output_path)?;
            stats.samples_written += 1;
        }
        stats.batches_processed += 1;
    }

    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
fn build_batch_request(
    samples: &[BatchSample],
    num_step: usize,
    guidance_scale: f32,
    t_shift: f32,
    audio_chunk_duration: f32,
    audio_chunk_threshold: f32,
    preprocess_prompt: bool,
    postprocess_output: bool,
    layer_penalty_factor: f32,
    position_temperature: f32,
    class_temperature: f32,
    denoise: bool,
) -> GenerationRequest {
    let mut request = GenerationRequest::new_text_only(samples[0].text.clone());
    request.texts = samples.iter().map(|sample| sample.text.clone()).collect();
    request.languages = samples
        .iter()
        .map(|sample| sample.language.clone())
        .collect();
    request.ref_audios = samples
        .iter()
        .map(|sample| {
            sample
                .ref_audio
                .as_ref()
                .map(|path| ReferenceAudioInput::from_path(path.display().to_string()))
        })
        .collect();
    request.ref_texts = samples
        .iter()
        .map(|sample| sample.ref_text.clone())
        .collect();
    request.instructs = samples
        .iter()
        .map(|sample| sample.instruct.clone())
        .collect();
    request.voice_clone_prompts = vec![None; samples.len()];
    request.speeds = samples.iter().map(|sample| sample.speed).collect();
    request.durations = samples.iter().map(|sample| sample.duration).collect();
    request.generation_config.num_step = num_step;
    request.generation_config.guidance_scale = guidance_scale;
    request.generation_config.t_shift = t_shift;
    request.generation_config.audio_chunk_duration = audio_chunk_duration;
    request.generation_config.audio_chunk_threshold = audio_chunk_threshold;
    request.generation_config.preprocess_prompt = preprocess_prompt;
    request.generation_config.postprocess_output = postprocess_output;
    request.generation_config.layer_penalty_factor = layer_penalty_factor;
    request.generation_config.position_temperature = position_temperature;
    request.generation_config.class_temperature = class_temperature;
    request.generation_config.denoise = denoise;
    request
}

fn read_test_list(
    path: &PathBuf,
    lang_id_fallback: Option<&str>,
) -> Result<Vec<BatchSample>, OmniVoiceError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some(id) = value.get("id").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(text) = value.get("text").and_then(|value| value.as_str()) else {
            return Err(OmniVoiceError::InvalidData(format!(
                "test list line {} is missing required field `text`",
                line_no + 1
            )));
        };
        let language = lang_id_fallback
            .map(str::to_string)
            .or_else(|| {
                value
                    .get("language_id")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            })
            .or_else(|| {
                value
                    .get("language_name")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            });
        samples.push(BatchSample {
            id: id.to_string(),
            text: text.to_string(),
            ref_audio: value
                .get("ref_audio")
                .and_then(|value| value.as_str())
                .map(PathBuf::from),
            ref_text: value
                .get("ref_text")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            instruct: value
                .get("instruct")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            language,
            duration: value
                .get("duration")
                .and_then(|value| value.as_f64())
                .map(|value| value as f32),
            speed: value
                .get("speed")
                .and_then(|value| value.as_f64())
                .map(|value| value as f32),
        });
    }
    Ok(samples)
}

fn cluster_samples_by_batch_size(
    samples: &[BatchSample],
    frontend: &Frontend,
    processor: &ReferenceAudioProcessor,
    batch_size: usize,
) -> Result<Vec<BatchJob>, OmniVoiceError> {
    if batch_size == 0 {
        return Err(OmniVoiceError::InvalidRequest(
            "--batch-size must be > 0".to_string(),
        ));
    }
    let mut samples_with_duration = samples
        .iter()
        .map(|sample| {
            estimate_sample_total_duration(frontend, processor, sample)
                .map(|duration| (sample.clone(), duration))
        })
        .collect::<Result<Vec<_>, _>>()?;
    samples_with_duration.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted = samples_with_duration
        .into_iter()
        .map(|(sample, _)| sample)
        .collect::<Vec<_>>();
    Ok(sorted
        .chunks(batch_size)
        .map(|chunk| BatchJob {
            samples: chunk.to_vec(),
        })
        .collect())
}

fn cluster_samples_by_duration(
    samples: &[BatchSample],
    frontend: &Frontend,
    processor: &ReferenceAudioProcessor,
    batch_duration: f32,
) -> Result<Vec<BatchJob>, OmniVoiceError> {
    let mut samples_with_duration = samples
        .iter()
        .map(|sample| {
            estimate_sample_total_duration(frontend, processor, sample)
                .map(|duration| (sample.clone(), duration))
        })
        .collect::<Result<Vec<_>, _>>()?;
    samples_with_duration.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut jobs = Vec::new();
    let mut current_samples = Vec::new();
    let mut current_duration = 0.0f32;
    for (sample, duration) in samples_with_duration {
        if duration > batch_duration {
            if !current_samples.is_empty() {
                jobs.push(BatchJob {
                    samples: std::mem::take(&mut current_samples),
                });
                current_duration = 0.0;
            }
            jobs.push(BatchJob {
                samples: vec![sample],
            });
            continue;
        }
        if current_samples.is_empty() || current_duration + duration <= batch_duration {
            current_duration += duration;
            current_samples.push(sample);
        } else {
            jobs.push(BatchJob {
                samples: std::mem::take(&mut current_samples),
            });
            current_duration = duration;
            current_samples.push(sample);
        }
    }
    if !current_samples.is_empty() {
        jobs.push(BatchJob {
            samples: current_samples,
        });
    }
    Ok(jobs)
}

fn estimate_sample_total_duration(
    frontend: &Frontend,
    processor: &ReferenceAudioProcessor,
    sample: &BatchSample,
) -> Result<f32, OmniVoiceError> {
    let ref_duration = if let Some(ref_audio) = &sample.ref_audio {
        let input = ReferenceAudioInput::from_path(ref_audio.display().to_string());
        let waveform = processor.load_input(&input)?;
        waveform.samples.len() as f32 / waveform.sample_rate as f32
    } else {
        0.0
    };
    let estimated_generation_seconds = if let Some(duration) = sample.duration {
        duration
    } else {
        let num_ref_audio_tokens = if ref_duration > 0.0 {
            Some((ref_duration * frontend.frame_rate() as f32).max(1.0) as usize)
        } else {
            None
        };
        frontend.estimate_target_tokens(
            &sample.text,
            sample.ref_text.as_deref(),
            num_ref_audio_tokens,
            sample.speed.unwrap_or(1.0),
        ) as f32
            / frontend.frame_rate() as f32
    };
    Ok(ref_duration + estimated_generation_seconds)
}

fn resolve_batch_devices(
    device: DeviceSpec,
    nj_per_gpu: usize,
) -> Result<Vec<DeviceSpec>, OmniVoiceError> {
    let base_devices = match device {
        DeviceSpec::Auto => {
            let cuda_devices = detect_cuda_devices();
            if !cuda_devices.is_empty() {
                cuda_devices
            } else if detect_metal_device_available() {
                vec![DeviceSpec::Metal]
            } else {
                vec![DeviceSpec::Cpu]
            }
        }
        explicit => vec![explicit],
    };
    let mut worker_devices = Vec::with_capacity(base_devices.len() * nj_per_gpu);
    for base_device in base_devices {
        for _ in 0..nj_per_gpu {
            worker_devices.push(base_device);
        }
    }
    Ok(worker_devices)
}

fn detect_cuda_devices() -> Vec<DeviceSpec> {
    #[cfg(feature = "cuda")]
    {
        let mut devices = Vec::new();
        for index in 0..16 {
            match Device::new_cuda(index) {
                Ok(device) => {
                    drop(device);
                    devices.push(DeviceSpec::Cuda(index));
                }
                Err(_) => break,
            }
        }
        devices
    }
    #[cfg(not(feature = "cuda"))]
    {
        Vec::new()
    }
}

fn describe_runtime_device(device: &Device) -> String {
    match device.location() {
        candle_core::DeviceLocation::Cpu => format!("{:?}", DeviceSpec::Cpu),
        candle_core::DeviceLocation::Cuda { gpu_id } => format!("{:?}", DeviceSpec::Cuda(gpu_id)),
        candle_core::DeviceLocation::Metal { gpu_id: _ } => format!("{:?}", DeviceSpec::Metal),
    }
}

fn detect_metal_device_available() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        Device::new_metal(0).is_ok()
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

fn run_prepare_prompt(
    model_dir: PathBuf,
    reference_root: PathBuf,
    case: String,
    device: DeviceSpec,
    dtype: DTypeSpec,
) -> Result<(), OmniVoiceError> {
    let model_dir = resolve_model_root(&model_dir)?;
    let options = RuntimeOptions::new(model_dir)
        .with_device(device)
        .with_dtype(dtype);
    let pipeline = Phase3Pipeline::from_options(options.clone())?;
    let prepared = pipeline.load_prepared_prompts_from_reference_case(&reference_root, &case)?;

    println!("phase_marker={}", workspace_phase_marker());
    println!("command=prepare-prompt");
    println!("case={case}");
    println!("model_root={}", options.model_root().display());
    println!("device={:?}", options.device());
    println!("dtype={:?}", options.dtype());
    println!(
        "resolved_device={}",
        describe_runtime_device(pipeline.stage0().device())
    );
    println!("resolved_dtype={:?}", pipeline.stage0().runtime_dtype());
    println!("kind={}", prepared.kind());
    println!("stage0_loaded={}", pipeline.stage0().is_loaded());
    println!("stage1_loaded={}", pipeline.stage1().is_loaded());

    match prepared {
        PreparedPromptSequence::Single(_) => {
            let batch = pipeline.prepare_prompt_from_reference_case(&reference_root, &case)?;
            println!("input_ids_dims={:?}", batch.input_ids_dims()?);
            println!("audio_mask_dims={:?}", batch.audio_mask_dims()?);
            println!("attention_mask_dims={:?}", batch.attention_mask_dims()?);
            println!("tokens_init_dims={:?}", batch.tokens_init_dims()?);
            println!("target_lens={:?}", batch.target_lens);
            println!("cond_lens={:?}", batch.cond_lens);
            println!("runtime_dtype={:?}", batch.runtime_dtype);
            println!(
                "num_audio_codebooks={}",
                pipeline.runtime_artifacts().contracts().num_audio_codebooks
            );
        }
        PreparedPromptSequence::Chunked(chunked) => {
            println!("chunk_count={}", chunked.prompts.len());
            println!("chunk_texts={:?}", chunked.chunk_texts);
            println!("chunk_target_lens={:?}", chunked.chunk_target_lens);
            for (index, prompt) in chunked.prompts.iter().enumerate() {
                println!("chunk[{index}].mode={:?}", prompt.mode);
                println!(
                    "chunk[{index}].input_ids_dims={:?}",
                    prompt.prompt.input_ids_dims()
                );
                println!(
                    "chunk[{index}].audio_mask_dims={:?}",
                    prompt.prompt.audio_mask_dims()
                );
                println!(
                    "chunk[{index}].target_start_idx={}",
                    prompt.target_start_idx
                );
                println!("chunk[{index}].total_length={}", prompt.total_length);
                println!("chunk[{index}].target_length={}", prompt.target_length);
                println!("chunk[{index}].style_text={}", prompt.style_text);
                println!("chunk[{index}].full_text={}", prompt.full_text);
            }
        }
    }

    Ok(())
}

fn run_stage1_prepare(
    model_dir: PathBuf,
    reference_root: PathBuf,
    case: String,
    device: DeviceSpec,
    dtype: DTypeSpec,
) -> Result<(), OmniVoiceError> {
    let model_dir = resolve_model_root(&model_dir)?;
    let options = RuntimeOptions::new(model_dir)
        .with_device(device)
        .with_dtype(dtype);
    let pipeline = Phase3Pipeline::from_options(options.clone())?;
    let decode = pipeline.prepare_stage1_from_reference_case(&reference_root, &case)?;

    println!("phase_marker={}", workspace_phase_marker());
    println!("command=stage1-prepare");
    println!("case={case}");
    println!("model_root={}", options.model_root().display());
    println!("device={:?}", options.device());
    println!("dtype={:?}", options.dtype());
    println!(
        "resolved_device={}",
        describe_runtime_device(pipeline.stage0().device())
    );
    println!("resolved_dtype={:?}", pipeline.stage0().runtime_dtype());
    println!("token_dims={:?}", decode.token_dims()?);
    println!("token_dtype={:?}", decode.tokens.dtype());
    println!("sample_rate={}", decode.sample_rate);
    println!("hop_length={}", decode.hop_length);
    println!("frame_rate={}", decode.frame_rate);
    println!("expected_codebooks={}", decode.expected_codebooks);
    println!(
        "ref_rms={}",
        decode
            .ref_rms
            .map(|value| value.to_string())
            .unwrap_or_else(|| "<none>".to_string())
    );
    println!("runtime_dtype={:?}", decode.runtime_dtype);
    println!("stage0_loaded={}", pipeline.stage0().is_loaded());
    println!("stage1_loaded={}", pipeline.stage1().is_loaded());

    Ok(())
}

fn run_stage1_decode(
    model_dir: PathBuf,
    reference_root: PathBuf,
    case: String,
    out: PathBuf,
    raw: bool,
    device: DeviceSpec,
    dtype: DTypeSpec,
) -> Result<(), OmniVoiceError> {
    let model_dir = resolve_model_root(&model_dir)?;
    let options = RuntimeOptions::new(model_dir)
        .with_device(device)
        .with_dtype(dtype);
    let pipeline = Phase3Pipeline::from_options(options.clone())?;
    let reference = ReferenceArtifactBundle::from_root(&reference_root)?;
    let reference_case = reference.case_by_id(&case)?;

    let decoded = if raw {
        pipeline.decode_stage1_raw_from_reference_case(&reference_root, &case)?
    } else {
        pipeline.decode_stage1_final_from_reference_case(&reference_root, &case)?
    };
    let expected = if raw {
        reference_case.load_decoded_raw_audio()?
    } else {
        reference_case.load_final_audio()?
    };
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    decoded.write_wav(&out)?;
    let metrics = decoded.parity_metrics(&expected)?;

    println!("phase_marker={}", workspace_phase_marker());
    println!("command=stage1-decode");
    println!("mode={}", if raw { "raw" } else { "final" });
    println!("case={case}");
    println!("model_root={}", options.model_root().display());
    println!("device={:?}", options.device());
    println!("dtype={:?}", options.dtype());
    println!(
        "resolved_device={}",
        describe_runtime_device(pipeline.stage0().device())
    );
    println!("resolved_dtype={:?}", pipeline.stage0().runtime_dtype());
    println!("out={}", out.display());
    println!("sample_rate={}", decoded.sample_rate);
    println!("frame_count={}", decoded.frame_count());
    println!("reference_frame_count={}", expected.frame_count());
    println!("max_abs={}", metrics.max_abs);
    println!("mae={}", metrics.mae);
    println!("rmse={}", metrics.rmse);
    println!("stage0_loaded={}", pipeline.stage0().is_loaded());
    println!("stage1_loaded={}", pipeline.stage1().is_loaded());

    Ok(())
}

fn run_stage0_generate(
    model_dir: PathBuf,
    reference_root: PathBuf,
    case: String,
    out: PathBuf,
    device: DeviceSpec,
    dtype: DTypeSpec,
) -> Result<(), OmniVoiceError> {
    let model_dir = resolve_model_root(&model_dir)?;
    let options = RuntimeOptions::new(model_dir)
        .with_device(device)
        .with_dtype(dtype);
    let pipeline = Phase3Pipeline::from_options(options.clone())?;
    let tokens = pipeline.generate_stage0_from_reference_case(&reference_root, &case)?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let payload = generated_tokens_json(&tokens);
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&payload).map_err(|error| {
            OmniVoiceError::InvalidData(format!("failed to serialize stage0 tokens: {error}"))
        })?,
    )?;

    println!("phase_marker={}", workspace_phase_marker());
    println!("command=stage0-generate");
    println!("case={case}");
    println!("model_root={}", options.model_root().display());
    println!("device={:?}", options.device());
    println!("dtype={:?}", options.dtype());
    println!(
        "resolved_device={}",
        describe_runtime_device(pipeline.stage0().device())
    );
    println!("resolved_dtype={:?}", pipeline.stage0().runtime_dtype());
    println!("out={}", out.display());
    println!("stage0_loaded={}", pipeline.stage0().is_loaded());
    println!("stage1_loaded={}", pipeline.stage1().is_loaded());
    match tokens {
        GeneratedTokens::Single(tokens) => {
            println!("kind=single");
            println!("token_dims={:?}", tokens.dims());
        }
        GeneratedTokens::Chunked(chunks) => {
            println!("kind=chunked");
            println!("chunk_count={}", chunks.len());
            for (index, chunk) in chunks.iter().enumerate() {
                println!("chunk[{index}].token_dims={:?}", chunk.dims());
            }
        }
    }
    Ok(())
}

fn run_stage0_debug(
    model_dir: PathBuf,
    reference_root: PathBuf,
    case: String,
    device: DeviceSpec,
    dtype: DTypeSpec,
) -> Result<(), OmniVoiceError> {
    let model_dir = resolve_model_root(&model_dir)?;
    let options = RuntimeOptions::new(model_dir)
        .with_device(device)
        .with_dtype(dtype);
    let pipeline = Phase3Pipeline::from_options(options.clone())?;
    let debug = pipeline.debug_stage0_from_reference_case(&reference_root, &case)?;

    println!("phase_marker={}", workspace_phase_marker());
    println!("command=stage0-debug");
    println!("case={case}");
    println!("model_root={}", options.model_root().display());
    println!("device={:?}", options.device());
    println!("dtype={:?}", options.dtype());
    println!(
        "resolved_device={}",
        describe_runtime_device(pipeline.stage0().device())
    );
    println!("resolved_dtype={:?}", pipeline.stage0().runtime_dtype());
    println!("token_dims={:?}", debug.tokens.dims());
    println!("stage0_loaded={}", pipeline.stage0().is_loaded());
    println!("stage1_loaded={}", pipeline.stage1().is_loaded());
    for (name, metric) in &debug.parity_metrics.metrics {
        println!(
            "metric.{name}=exact:{} max_abs:{} mae:{} rmse:{}",
            metric.exact_match, metric.max_abs, metric.mae, metric.rmse
        );
    }
    Ok(())
}

fn join_paths(values: &BTreeSet<String>) -> String {
    values.iter().cloned().collect::<Vec<_>>().join(",")
}

fn model_arg_or_default(value: Option<PathBuf>) -> PathBuf {
    value.unwrap_or_else(|| PathBuf::from(DEFAULT_OMNIVOICE_REPO))
}

fn required_path_arg(value: Option<PathBuf>, name: &str) -> Result<PathBuf, OmniVoiceError> {
    value.ok_or_else(|| {
        OmniVoiceError::InvalidRequest(format!("missing required {name}\n{}", usage()))
    })
}

fn required_string_arg(value: Option<String>, name: &str) -> Result<String, OmniVoiceError> {
    value.ok_or_else(|| {
        OmniVoiceError::InvalidRequest(format!("missing required {name}\n{}", usage()))
    })
}

fn generated_tokens_json(tokens: &GeneratedTokens) -> serde_json::Value {
    match tokens {
        GeneratedTokens::Single(tokens) => serde_json::json!({
            "kind": "single",
            "dims": tokens.dims(),
            "tokens": tokens.data,
        }),
        GeneratedTokens::Chunked(chunks) => serde_json::json!({
            "kind": "chunked",
            "chunks": chunks.iter().enumerate().map(|(index, chunk)| serde_json::json!({
                "index": index,
                "dims": chunk.dims(),
                "tokens": chunk.data,
            })).collect::<Vec<_>>(),
        }),
    }
}

fn required_value<'a>(
    args: &'a [String],
    index: usize,
    name: &str,
) -> Result<&'a str, OmniVoiceError> {
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| OmniVoiceError::InvalidRequest(format!("{name} requires a value")))
}

fn parse_usize_arg(args: &[String], index: usize, name: &str) -> Result<usize, OmniVoiceError> {
    required_value(args, index, name)?
        .parse::<usize>()
        .map_err(|_| OmniVoiceError::InvalidRequest(format!("{name} requires an integer value")))
}

fn parse_f32_arg(args: &[String], index: usize, name: &str) -> Result<f32, OmniVoiceError> {
    required_value(args, index, name)?
        .parse::<f32>()
        .map_err(|_| OmniVoiceError::InvalidRequest(format!("{name} requires a float value")))
}

fn parse_u64_arg(args: &[String], index: usize, name: &str) -> Result<u64, OmniVoiceError> {
    required_value(args, index, name)?
        .parse::<u64>()
        .map_err(|_| OmniVoiceError::InvalidRequest(format!("{name} requires an integer value")))
}

fn parse_bool_arg(args: &[String], index: usize, name: &str) -> Result<bool, OmniVoiceError> {
    match required_value(args, index, name)?
        .to_ascii_lowercase()
        .as_str()
    {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(OmniVoiceError::InvalidRequest(format!(
            "{name} requires true/false"
        ))),
    }
}

fn device_usage_values() -> &'static str {
    "auto|cuda|cuda:N|mps|metal|cpu"
}

fn artifacts_validate_usage(program: &str) -> String {
    format!("usage:\n  {program} validate [--model <path-or-hf-repo>] [--reference-root <path>]")
}

fn infer_usage(program: &str) -> String {
    format!(
        concat!(
            "usage:\n  {program} [--model <path-or-hf-repo>] --text <text> --output <wav> ",
            "[--language <lang>] [--ref_audio <wav>] [--ref_text <text>] [--instruct <text>] ",
            "[--duration <seconds>] [--speed <factor>] [--asr-model <path-or-hf-repo>] ",
            "[--device {device}] [--dtype auto|f16|bf16|f32] [--num_step <n>] ",
            "[--guidance_scale <value>] [--t_shift <value>] [--layer_penalty_factor <value>] ",
            "[--position_temperature <value>] [--class_temperature <value>] ",
            "[--preprocess_prompt <bool>] [--postprocess_output <bool>] [--denoise <bool>] ",
            "[--audio_chunk_duration <seconds>] [--audio_chunk_threshold <seconds>] [--seed <u64>]\n\n",
            "aliases:\n  --ref-audio, --ref-text, --num-step, --guidance-scale, --t-shift,\n",
            "  --layer-penalty-factor, --position-temperature, --class-temperature,\n",
            "  --preprocess-prompt, --postprocess-output, --audio-chunk-duration,\n",
            "  --audio-chunk-threshold"
        ),
        program = program,
        device = device_usage_values(),
    )
}

fn infer_batch_usage(program: &str) -> String {
    format!(
        concat!(
            "usage:\n  {program} [--model <path-or-hf-repo>] --test_list <jsonl> --res_dir <dir> ",
            "[--device {device}] [--dtype auto|f16|bf16|f32] [--num_step <n>] ",
            "[--guidance_scale <value>] [--t_shift <value>] [--nj_per_gpu <n>] ",
            "[--audio_chunk_duration <seconds>] [--audio_chunk_threshold <seconds>] ",
            "[--batch_duration <seconds>] [--batch_size <n>] [--warmup <n>] ",
            "[--preprocess_prompt <bool>] [--postprocess_output <bool>] ",
            "[--layer_penalty_factor <value>] [--position_temperature <value>] ",
            "[--class_temperature <value>] [--denoise <bool>] [--lang_id <lang>] [--seed <u64>]\n\n",
            "aliases:\n  --test-list, --res-dir, --num-step, --guidance-scale, --t-shift,\n",
            "  --nj-per-gpu, --audio-chunk-duration, --audio-chunk-threshold,\n",
            "  --batch-duration, --batch-size, --preprocess-prompt,\n",
            "  --postprocess-output, --layer-penalty-factor,\n",
            "  --position-temperature, --class-temperature, --lang-id"
        ),
        program = program,
        device = device_usage_values(),
    )
}

fn prepare_prompt_usage(program: &str) -> String {
    format!(
        "usage:\n  {program} [--model <path-or-hf-repo>] --reference-root <path> --case <id> [--device {}] [--dtype auto|f16|bf16|f32]",
        device_usage_values()
    )
}

fn stage1_prepare_usage(program: &str) -> String {
    format!(
        "usage:\n  {program} [--model <path-or-hf-repo>] --reference-root <path> --case <id> [--device {}] [--dtype auto|f16|bf16|f32]",
        device_usage_values()
    )
}

fn stage1_decode_usage(program: &str) -> String {
    format!(
        "usage:\n  {program} [--model <path-or-hf-repo>] --reference-root <path> --case <id> --out <wav> [--raw] [--device {}] [--dtype auto|f16|bf16|f32]",
        device_usage_values()
    )
}

fn stage0_generate_usage(program: &str) -> String {
    format!(
        "usage:\n  {program} [--model <path-or-hf-repo>] --reference-root <path> --case <id> --out <json> [--device {}] [--dtype auto|f16|bf16|f32]",
        device_usage_values()
    )
}

fn stage0_debug_usage(program: &str) -> String {
    format!(
        "usage:\n  {program} [--model <path-or-hf-repo>] --reference-root <path> --case <id> [--device {}] [--dtype auto|f16|bf16|f32]",
        device_usage_values()
    )
}

fn usage() -> String {
    [
        "usage:",
        "  omnivoice-cli artifacts validate [--model <path-or-hf-repo>] [--reference-root <path>]",
        "  omnivoice-cli infer [--model <path-or-hf-repo>] --text <text> --output <wav> [--language <lang>] [--ref-audio <wav>] [--ref-text <text>] [--instruct <text>] [--duration <seconds>] [--speed <factor>] [--asr-model <path-or-hf-repo>] [--device auto|cuda|cuda:N|mps|metal|cpu] [--dtype auto|f16|bf16|f32] [--seed <u64>]",
        "  omnivoice-cli infer-batch [--model <path-or-hf-repo>] --test-list <jsonl> --res-dir <dir> [--device auto|cuda|cuda:N|mps|metal|cpu] [--dtype auto|f16|bf16|f32] [--batch-size <n>] [--batch-duration <seconds>] [--nj-per-gpu <n>] [--warmup <n>]",
        "  omnivoice-cli prepare-prompt [--model <path-or-hf-repo>] --reference-root <path> --case <id> [--device auto|cuda|cuda:N|mps|metal|cpu] [--dtype auto|f16|bf16|f32]",
        "  omnivoice-cli stage1-prepare [--model <path-or-hf-repo>] --reference-root <path> --case <id> [--device auto|cuda|cuda:N|mps|metal|cpu] [--dtype auto|f16|bf16|f32]",
        "  omnivoice-cli stage1-decode [--model <path-or-hf-repo>] --reference-root <path> --case <id> --out <wav> [--raw] [--device auto|cuda|cuda:N|mps|metal|cpu] [--dtype auto|f16|bf16|f32]",
        "  omnivoice-cli stage0-generate [--model <path-or-hf-repo>] --reference-root <path> --case <id> --out <json> [--device auto|cuda|cuda:N|mps|metal|cpu] [--dtype auto|f16|bf16|f32]",
        "  omnivoice-cli stage0-debug [--model <path-or-hf-repo>] --reference-root <path> --case <id> [--device auto|cuda|cuda:N|mps|metal|cpu] [--dtype auto|f16|bf16|f32]",
    ]
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::{parse_entrypoint_command, requested_help, CliCommand, InvocationMode};
    use omnivoice_infer::{
        artifacts::RuntimeArtifactManifest, model_source::manifest_download_targets, DeviceSpec,
    };
    use std::path::PathBuf;

    #[test]
    fn infer_defaults_to_official_repo_when_model_is_omitted() {
        let command = CliCommand::parse(&[
            "infer".to_string(),
            "--text".to_string(),
            "hello".to_string(),
            "--output".to_string(),
            "out.wav".to_string(),
        ])
        .unwrap();

        match command {
            CliCommand::Infer { model_dir, .. } => {
                assert_eq!(model_dir, PathBuf::from("k2-fsa/OmniVoice"));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn artifacts_validate_defaults_to_official_repo_when_model_is_omitted() {
        let command =
            CliCommand::parse(&["artifacts".to_string(), "validate".to_string()]).unwrap();

        match command {
            CliCommand::ArtifactsValidate { model_dir, .. } => {
                assert_eq!(model_dir, PathBuf::from("k2-fsa/OmniVoice"));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn manifest_download_targets_are_manifest_scoped_and_deduplicated() {
        let manifest: RuntimeArtifactManifest = serde_json::from_str(
            r#"{
                "version": 1,
                "generator": {
                    "config": "config.json",
                    "weights": "model.safetensors",
                    "required_prefixes": ["llm"],
                    "ignored_keys": []
                },
                "text_tokenizer": {
                    "tokenizer": "tokenizer.json",
                    "tokenizer_config": "tokenizer_config.json",
                    "metadata": {
                        "chat_template": "chat_template.jinja"
                    }
                },
                "audio_tokenizer": {
                    "config": "audio_tokenizer/config.json",
                    "weights": "audio_tokenizer/model.safetensors",
                    "preprocessor_config": "audio_tokenizer/preprocessor_config.json",
                    "required_prefixes": ["quantizer"],
                    "metadata": {
                        "license": "audio_tokenizer/LICENSE"
                    }
                },
                "contracts": {
                    "num_audio_codebooks": 8,
                    "audio_vocab_size": 1025,
                    "audio_mask_id": 1024,
                    "token_id_min": 0,
                    "token_id_max": 1023,
                    "sample_rate": 24000,
                    "hop_length": 960,
                    "frame_rate": 25
                }
            }"#,
        )
        .unwrap();

        let targets = manifest_download_targets(&manifest);

        assert_eq!(
            targets,
            vec![
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "audio_tokenizer/config.json",
                "audio_tokenizer/model.safetensors",
                "audio_tokenizer/preprocessor_config.json",
                "audio_tokenizer/LICENSE",
            ]
        );
    }

    #[test]
    fn infer_wrapper_help_uses_wrapper_program_name() {
        let help = requested_help(InvocationMode::InferWrapper, &["--help".to_string()])
            .expect("wrapper help should be available");

        assert!(help.contains("omnivoice-infer"));
        assert!(!help.contains("omnivoice-cli infer ["));
    }

    #[test]
    fn infer_subcommand_help_uses_subcommand_program_name() {
        let help = requested_help(
            InvocationMode::Multi,
            &["infer".to_string(), "--help".to_string()],
        )
        .expect("subcommand help should be available");

        assert!(help.contains("omnivoice-cli infer"));
        assert!(!help.contains("omnivoice-infer-batch"));
    }

    #[test]
    fn infer_wrapper_accepts_official_underscore_aliases() {
        let command = parse_entrypoint_command(
            InvocationMode::InferWrapper,
            &[
                "--model".to_string(),
                "checkpoint".to_string(),
                "--text".to_string(),
                "hello".to_string(),
                "--output".to_string(),
                "out.wav".to_string(),
                "--ref_audio".to_string(),
                "ref.wav".to_string(),
                "--ref_text".to_string(),
                "reference".to_string(),
                "--num_step".to_string(),
                "16".to_string(),
                "--guidance_scale".to_string(),
                "2.5".to_string(),
                "--t_shift".to_string(),
                "0.2".to_string(),
                "--layer_penalty_factor".to_string(),
                "4.0".to_string(),
                "--position_temperature".to_string(),
                "1.0".to_string(),
                "--class_temperature".to_string(),
                "0.5".to_string(),
                "--postprocess_output".to_string(),
                "false".to_string(),
                "--preprocess_prompt".to_string(),
                "false".to_string(),
                "--audio_chunk_duration".to_string(),
                "12.0".to_string(),
                "--audio_chunk_threshold".to_string(),
                "24.0".to_string(),
                "--device".to_string(),
                "cuda".to_string(),
            ],
        )
        .unwrap();

        match command {
            CliCommand::Infer {
                model_dir,
                ref_audio,
                ref_text,
                num_step,
                guidance_scale,
                t_shift,
                layer_penalty_factor,
                position_temperature,
                class_temperature,
                postprocess_output,
                preprocess_prompt,
                audio_chunk_duration,
                audio_chunk_threshold,
                device,
                ..
            } => {
                assert_eq!(model_dir, PathBuf::from("checkpoint"));
                assert_eq!(ref_audio, Some(PathBuf::from("ref.wav")));
                assert_eq!(ref_text.as_deref(), Some("reference"));
                assert_eq!(num_step, 16);
                assert_eq!(guidance_scale, 2.5);
                assert_eq!(t_shift, 0.2);
                assert_eq!(layer_penalty_factor, 4.0);
                assert_eq!(position_temperature, 1.0);
                assert_eq!(class_temperature, 0.5);
                assert!(!postprocess_output);
                assert!(!preprocess_prompt);
                assert_eq!(audio_chunk_duration, 12.0);
                assert_eq!(audio_chunk_threshold, 24.0);
                assert_eq!(device, DeviceSpec::Cuda(0));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn infer_batch_wrapper_accepts_official_underscore_aliases() {
        let command = parse_entrypoint_command(
            InvocationMode::InferBatchWrapper,
            &[
                "--model".to_string(),
                "checkpoint".to_string(),
                "--test_list".to_string(),
                "cases.jsonl".to_string(),
                "--res_dir".to_string(),
                "results".to_string(),
                "--num_step".to_string(),
                "8".to_string(),
                "--guidance_scale".to_string(),
                "1.5".to_string(),
                "--t_shift".to_string(),
                "0.3".to_string(),
                "--nj_per_gpu".to_string(),
                "2".to_string(),
                "--audio_chunk_duration".to_string(),
                "10.0".to_string(),
                "--audio_chunk_threshold".to_string(),
                "20.0".to_string(),
                "--batch_duration".to_string(),
                "60.0".to_string(),
                "--batch_size".to_string(),
                "4".to_string(),
                "--preprocess_prompt".to_string(),
                "false".to_string(),
                "--postprocess_output".to_string(),
                "false".to_string(),
                "--layer_penalty_factor".to_string(),
                "4.0".to_string(),
                "--position_temperature".to_string(),
                "1.0".to_string(),
                "--class_temperature".to_string(),
                "0.25".to_string(),
                "--lang_id".to_string(),
                "en".to_string(),
                "--device".to_string(),
                "mps".to_string(),
            ],
        )
        .unwrap();

        match command {
            CliCommand::InferBatch {
                model_dir,
                test_list,
                res_dir,
                num_step,
                guidance_scale,
                t_shift,
                nj_per_gpu,
                audio_chunk_duration,
                audio_chunk_threshold,
                batch_duration,
                batch_size,
                preprocess_prompt,
                postprocess_output,
                layer_penalty_factor,
                position_temperature,
                class_temperature,
                lang_id,
                device,
                ..
            } => {
                assert_eq!(model_dir, PathBuf::from("checkpoint"));
                assert_eq!(test_list, PathBuf::from("cases.jsonl"));
                assert_eq!(res_dir, PathBuf::from("results"));
                assert_eq!(num_step, 8);
                assert_eq!(guidance_scale, 1.5);
                assert_eq!(t_shift, 0.3);
                assert_eq!(nj_per_gpu, 2);
                assert_eq!(audio_chunk_duration, 10.0);
                assert_eq!(audio_chunk_threshold, 20.0);
                assert_eq!(batch_duration, 60.0);
                assert_eq!(batch_size, 4);
                assert!(!preprocess_prompt);
                assert!(!postprocess_output);
                assert_eq!(layer_penalty_factor, 4.0);
                assert_eq!(position_temperature, 1.0);
                assert_eq!(class_temperature, 0.25);
                assert_eq!(lang_id.as_deref(), Some("en"));
                assert_eq!(device, DeviceSpec::Metal);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }
}
