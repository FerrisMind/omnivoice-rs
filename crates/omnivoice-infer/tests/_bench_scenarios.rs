
use omnivoice_infer::contracts::{GenerationRequest, ReferenceAudioInput};
use omnivoice_infer::pipeline::Phase3Pipeline;
use omnivoice_infer::runtime::{DTypeSpec, DeviceSpec, RuntimeOptions};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct Scenario {
    id: String,
    title: String,
    mode: String,
    text: String,
    language: Option<String>,
    instruct: Option<String>,
    use_ref_audio: bool,
    use_ref_text: bool,
    duration: Option<f32>,
    speed: Option<f32>,
    #[serde(default)]
    use_cached_prompt: bool,
}

#[derive(Debug, Deserialize)]
struct BenchConfig {
    model: String,
    ref_audio: String,
    ref_text: String,
    device: String,
    #[serde(default = "default_dtype")]
    dtype: String,
    seed: u64,
    repeats: usize,
    scenarios: Vec<Scenario>,
}

fn default_dtype() -> String {
    "f16".to_string()
}

fn parse_device(s: &str) -> DeviceSpec {
    if s == "cpu" {
        DeviceSpec::Cpu
    } else if s == "cuda" || s == "cuda:0" {
        DeviceSpec::Cuda(0)
    } else if let Some(rest) = s.strip_prefix("cuda:") {
        DeviceSpec::Cuda(rest.parse().expect("cuda index"))
    } else {
        panic!("unsupported device {s}");
    }
}

#[test]
fn bench_listening_scenarios() {
    let path = std::env::var("OMNIVOICE_BENCH_CONFIG").expect("OMNIVOICE_BENCH_CONFIG");
    let cfg: BenchConfig =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).expect("config json");
    let model = PathBuf::from(&cfg.model);
    let device = parse_device(&cfg.device);
    let dtype = match cfg.dtype.to_ascii_lowercase().as_str() {
        "f32" | "float32" => DTypeSpec::F32,
        "f16" | "float16" => DTypeSpec::F16,
        "bf16" | "bfloat16" => DTypeSpec::BF16,
        other => panic!("unsupported dtype {other}"),
    };

    let t_load = Instant::now();
    let pipe = Phase3Pipeline::from_options(
        RuntimeOptions::new(&model)
            .with_device(device)
            .with_dtype(dtype)
            .with_seed(cfg.seed),
    )
    .expect("pipeline");
    let load_s = t_load.elapsed().as_secs_f64();

    // warmup: text + clone so audio-tokenizer weights are loaded before timing
    let mut warm = GenerationRequest::new_text_only("Warmup.".to_string());
    warm.languages = vec![Some("English".to_string())];
    warm.generation_config.num_step = 8;
    let _ = pipe.generate(&warm);
    let mut warm_clone = GenerationRequest::new_text_only("Warmup clone.".to_string());
    warm_clone.languages = vec![Some("English".to_string())];
    warm_clone.generation_config.num_step = 8;
    warm_clone.ref_audios = vec![Some(ReferenceAudioInput::from_path(cfg.ref_audio.clone()))];
    warm_clone.ref_texts = vec![Some(cfg.ref_text.clone())];
    let _ = pipe.generate(&warm_clone);

    let mut rows = Vec::new();
    for sc in &cfg.scenarios {
        let mut times = Vec::new();
        let mut audio_s = None;
        let mut target_tokens: Option<usize> = None;
        let mut err: Option<String> = None;
        for rep in 0..cfg.repeats.max(1) {
            let mut req = GenerationRequest::new_text_only(sc.text.clone());
            req.languages = vec![sc.language.clone()];
            req.instructs = vec![sc.instruct.clone()];
            req.durations = vec![sc.duration];
            req.speeds = vec![sc.speed];
            req.generation_config.num_step = 32;
            req.generation_config.guidance_scale = 2.0;
            req.generation_config.t_shift = 0.1;
            req.generation_config.denoise = true;
            req.generation_config.position_temperature = 5.0;
            req.generation_config.class_temperature = 0.0;
            req.generation_config.layer_penalty_factor = 5.0;
            req.generation_config.postprocess_output = true;

            if sc.use_ref_audio {
                if sc.use_cached_prompt && sc.use_ref_text {
                    match pipe.create_voice_clone_prompt_from_audio(
                        &ReferenceAudioInput::from_path(cfg.ref_audio.clone()),
                        Some(cfg.ref_text.as_str()),
                        true,
                        None,
                    ) {
                        Ok(prompt) => {
                            req.voice_clone_prompts = vec![Some(prompt)];
                            req.ref_audios = vec![None];
                            req.ref_texts = vec![None];
                        }
                        Err(e) => {
                            err = Some(format!("clone prompt: {e}"));
                            break;
                        }
                    }
                } else {
                    req.ref_audios = vec![Some(ReferenceAudioInput::from_path(cfg.ref_audio.clone()))];
                    req.ref_texts = vec![if sc.use_ref_text {
                        Some(cfg.ref_text.clone())
                    } else {
                        None
                    }];
                }
            }

            // re-seed between reps for comparable noise streams
            let _ = pipe.stage0().set_seed(cfg.seed + rep as u64);

            let t0 = Instant::now();
            match pipe.generate_with_usage(&req) {
                Ok(results) => {
                    let elapsed = t0.elapsed().as_secs_f64();
                    times.push(elapsed);
                    let n = results[0].audio.samples.len();
                    let sr = results[0].audio.sample_rate.max(1) as f64;
                    audio_s = Some(n as f64 / sr);
                    target_tokens = Some(results[0].usage.output_tokens);
                }
                Err(e) => {
                    err = Some(e.to_string());
                    break;
                }
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let wall = if times.is_empty() {
            None
        } else {
            Some(times[times.len() / 2])
        };
        let rtf = match (wall, audio_s) {
            (Some(w), Some(a)) if a > 0.0 => Some(w / a),
            _ => None,
        };
        rows.push(json!({
            "scenario_id": sc.id,
            "title": sc.title,
            "mode": sc.mode,
            "ok": err.is_none() && wall.is_some(),
            "error": err,
            "wall_s": wall,
            "wall_s_all": times,
            "audio_s": audio_s,
            "target_tokens": target_tokens,
            "rtf": rtf,
        }));
    }

    let out = json!({
        "load_s": load_s,
        "rows": rows,
    });
    println!("===BENCH_JSON_BEGIN===");
    println!("{}", serde_json::to_string(&out).unwrap());
    println!("===BENCH_JSON_END===");
}
