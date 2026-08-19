use std::collections::BTreeSet;
use std::path::PathBuf;

use omnivoice_infer::{
    artifacts::RuntimeArtifacts,
    audio_tokenizer::AudioTokenizerRuntimePlan,
    contracts::{GeneratedTokens, GenerationRequest},
    frontend::Frontend,
    pipeline::Phase3Pipeline,
    runtime::{DeviceSpec, RuntimeOptions},
    stage1_decoder::Stage1RuntimePlan,
};

fn gguf_root() -> PathBuf {
    PathBuf::from(r"C:\Users\PC\Documents\models\OmniVoice-GGUF")
}

fn gguf_available() -> bool {
    let root = gguf_root();
    root.join("omnivoice-base-BF16.gguf").is_file()
        && root.join("omnivoice-tokenizer-BF16.gguf").is_file()
}

fn sine(sr: u32, secs: f32, hz: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr as f32).sin() * 0.2)
        .collect()
}

fn zcr(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crosses = samples
        .windows(2)
        .filter(|w| w[0] * w[1] < 0.0)
        .count();
    crosses as f32 / (samples.len() - 1) as f32
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}

fn sign_hist(samples: &[f32]) -> (usize, usize, usize) {
    let mut neg = 0;
    let mut zero = 0;
    let mut pos = 0;
    for sample in samples {
        if *sample < 0.0 {
            neg += 1;
        } else if *sample > 0.0 {
            pos += 1;
        } else {
            zero += 1;
        }
    }
    (neg, zero, pos)
}

#[test]
fn gguf_prompt_special_token_ids() {
    if !gguf_available() {
        return;
    }
    let runtime = RuntimeArtifacts::from_model_root(gguf_root()).unwrap();
    let native = runtime.native_weight_dtype().unwrap();
    let lm_name = runtime
        .gguf_bundle()
        .unwrap()
        .generator_path()
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_uppercase();
    println!("gguf lm={lm_name} weight_dtype={native:?}");
    if lm_name.contains("Q4") || lm_name.contains("Q8") || lm_name.contains("Q5") {
        assert_eq!(native, omnivoice_infer::ModelWeightDType::Quantized);
    } else if lm_name.contains("BF16") {
        assert_eq!(native, omnivoice_infer::ModelWeightDType::BF16);
    } else if lm_name.contains("F16") {
        assert_eq!(native, omnivoice_infer::ModelWeightDType::F16);
    } else {
        assert_eq!(native, omnivoice_infer::ModelWeightDType::F32);
    }
    let frontend = Frontend::from_runtime_artifacts(&runtime).unwrap();
    let prepared = frontend
        .prepare_prompt(
            &frontend
                .build_task(&GenerationRequest::new_text_only("test"))
                .unwrap(),
            0,
        )
        .unwrap();
    println!("style_text={}", prepared.style_text);
    println!(
        "style_token_ids len={} ids={:?}",
        prepared.style_token_ids.len(),
        prepared.style_token_ids
    );
    println!("full_text={}", prepared.full_text);
    println!(
        "text_token_ids len={} ids={:?}",
        prepared.text_token_ids.len(),
        prepared.text_token_ids
    );
    println!(
        "target_start_idx={} total_length={} target_length={}",
        prepared.target_start_idx, prepared.total_length, prepared.target_length
    );
    assert_eq!(
        prepared.style_token_ids.len(),
        6,
        "OmniVoice specials must encode as atomic tokens, got {:?}",
        prepared.style_token_ids
    );
    assert!(
        prepared.text_token_ids.len() <= 4,
        "text specials must stay atomic, got {:?}",
        prepared.text_token_ids
    );
    assert!(
        prepared.target_start_idx < 16,
        "split specials inflate the prompt to {}",
        prepared.target_start_idx
    );
}

fn load_codec_plans() -> (
    RuntimeArtifacts,
    RuntimeOptions,
    AudioTokenizerRuntimePlan,
    Stage1RuntimePlan,
) {
    let options = RuntimeOptions::new(gguf_root()).with_device(DeviceSpec::Cpu);
    let runtime = RuntimeArtifacts::from_model_root(gguf_root()).unwrap();
    let tokenizer = AudioTokenizerRuntimePlan::from_runtime_artifacts_with_device(
        options.clone(),
        &runtime,
        candle_core::Device::Cpu,
    )
    .unwrap();
    let stage1 = Stage1RuntimePlan::from_runtime_artifacts_with_device(
        options.clone(),
        &runtime,
        omnivoice_infer::stage1_decoder::Stage1DecoderBundle::from_runtime_artifacts(&runtime)
            .unwrap(),
        candle_core::Device::Cpu,
    )
    .unwrap();
    (runtime, options, tokenizer, stage1)
}

#[test]
fn gguf_codec_sine_roundtrip() {
    if !gguf_available() {
        return;
    }
    let (runtime, _options, tokenizer, stage1) = load_codec_plans();
    let sr = runtime.contracts().sample_rate;
    let samples = sine(sr, 1.0, 220.0);
    println!(
        "input sine n={} rms={:.5} zcr={:.4}",
        samples.len(),
        rms(&samples),
        zcr(&samples)
    );
    let codes = tokenizer.encode_waveform(&samples, sr).unwrap();
    let unique: BTreeSet<i64> = codes.data.iter().copied().collect();
    println!(
        "encode codes dims={:?} unique={} min={} max={}",
        codes.dims(),
        unique.len(),
        codes.data.iter().copied().min().unwrap_or(0),
        codes.data.iter().copied().max().unwrap_or(0)
    );
    let (codebooks, frames) = codes.dims();
    for codebook in 0..codebooks {
        let start = codebook * frames;
        let row = &codes.data[start..start + frames];
        let row_unique: BTreeSet<i64> = row.iter().copied().collect();
        println!(
            "  codebook {codebook} unique={} min={} max={} preview={:?}",
            row_unique.len(),
            row.iter().copied().min().unwrap_or(0),
            row.iter().copied().max().unwrap_or(0),
            &row[..row.len().min(8)]
        );
    }

    let roundtrip = stage1
        .decode_raw(&GeneratedTokens::Single(codes), None)
        .unwrap();
    let (neg, zero, pos) = sign_hist(&roundtrip.samples);
    println!(
        "codec roundtrip n={} rms={:.5} zcr={:.4} signs neg/zero/pos={}/{}/{}",
        roundtrip.samples.len(),
        rms(&roundtrip.samples),
        zcr(&roundtrip.samples),
        neg,
        zero,
        pos
    );
}

#[test]
fn gguf_generated_token_stats() {
    if !gguf_available() {
        return;
    }
    let options = RuntimeOptions::new(gguf_root()).with_device(DeviceSpec::Cpu);
    let runtime = RuntimeArtifacts::from_model_root(gguf_root()).unwrap();
    let pipeline = Phase3Pipeline::from_options(options.clone()).unwrap();
    let tokens = pipeline
        .generate_tokens(&GenerationRequest::new_text_only("test"))
        .unwrap();
    let GeneratedTokens::Single(generated) = &tokens[0] else {
        panic!("expected single token tensor");
    };
    let unique: BTreeSet<i64> = generated.data.iter().copied().collect();
    let mask_id = runtime.contracts().audio_mask_id;
    let mask_count = generated
        .data
        .iter()
        .filter(|token| **token == mask_id)
        .count();
    println!(
        "generated dims={:?} unique={} min={} max={} mask_id={} mask_count={}",
        generated.dims(),
        unique.len(),
        generated.data.iter().copied().min().unwrap_or(0),
        generated.data.iter().copied().max().unwrap_or(0),
        mask_id,
        mask_count
    );
    let (codebooks, frames) = generated.dims();
    for codebook in 0..codebooks {
        let start = codebook * frames;
        let row = &generated.data[start..start + frames];
        let row_unique: BTreeSet<i64> = row.iter().copied().collect();
        println!(
            "  codebook {codebook} unique={} min={} max={} preview={:?}",
            row_unique.len(),
            row.iter().copied().min().unwrap_or(0),
            row.iter().copied().max().unwrap_or(0),
            &row[..row.len().min(16)]
        );
    }

    let stage1 = Stage1RuntimePlan::from_runtime_artifacts_with_device(
        options,
        &runtime,
        omnivoice_infer::stage1_decoder::Stage1DecoderBundle::from_runtime_artifacts(&runtime)
            .unwrap(),
        candle_core::Device::Cpu,
    )
    .unwrap();
    let decoded = stage1
        .decode_raw(&GeneratedTokens::Single(generated.clone()), None)
        .unwrap();
    let (neg, zero, pos) = sign_hist(&decoded.samples);
    let decode_zcr = zcr(&decoded.samples);
    println!(
        "generated decode n={} rms={:.5} zcr={:.4} signs neg/zero/pos={}/{}/{}",
        decoded.samples.len(),
        rms(&decoded.samples),
        decode_zcr,
        neg,
        zero,
        pos
    );
    let n = decoded.samples.len() as f32;
    assert!(
        decode_zcr > 0.05,
        "generated waveform collapsed to DC/rumble, zcr={decode_zcr}"
    );
    assert!(
        (neg as f32 / n) > 0.2 && (pos as f32 / n) > 0.2,
        "generated waveform is not bipolar, signs={neg}/{zero}/{pos}"
    );
}
