#!/usr/bin/env python3
"""Benchmark all 15 listening scenarios: Python ref vs Rust port × CUDA/CPU.

Outputs:
  artifacts/bench_scenarios/results.json
  artifacts/bench_scenarios/README.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "v2" / "model"
DEFAULT_REF_AUDIO = ROOT / "artifacts" / "v2" / "ref.wav"
DEFAULT_REF_TEXT = ROOT / "artifacts" / "v2" / "ref_text.txt"
DEFAULT_OUT = ROOT / "artifacts" / "bench_scenarios"
DEFAULT_SEED = 1234
SAMPLE_RATE = 24_000


@dataclass
class Scenario:
    id: str
    title: str
    mode: str
    text: str
    language: str | None = None
    instruct: str | None = None
    use_ref_audio: bool = False
    use_ref_text: bool = False
    duration: float | None = None
    speed: float | None = None
    use_cached_prompt: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_scenarios() -> list[Scenario]:
    long_text = (
        "OmniVoice can synthesize long-form speech while keeping memory usage stable. "
        "In this listening check, the text is intentionally longer than a typical demo "
        "sentence so the pipeline is forced through its chunking path. Each section "
        "should stay intelligible, maintain similar pacing, and join smoothly with the "
        "following section. We repeat the core idea so the model has enough text to split. "
        "OmniVoice can synthesize long-form speech while keeping memory usage stable."
    )
    return [
        Scenario("01_auto_en_short", "Auto EN short", "auto", "OmniVoice creates clear speech from text with minimal setup.", "English"),
        Scenario("02_auto_zh_short", "Auto ZH short", "auto", "欢迎使用 OmniVoice。这是一段中文自动音色的试听样本。", "Chinese"),
        Scenario("03_auto_lang_agnostic", "Auto no language", "auto", "This sample leaves language unspecified so the model chooses freely.", None),
        Scenario(
            "04_design_en_british_female",
            "Design British female",
            "design",
            "Good afternoon. This reference should sound calm, precise, and suitable for a polished product demo.",
            "English",
            instruct="female, low pitch, british accent",
        ),
        Scenario(
            "05_design_en_male_american",
            "Design male American",
            "design",
            "Hey there! Ready to ship another release? Let's make this one count.",
            "English",
            instruct="male, high pitch, young adult, american accent",
        ),
        Scenario(
            "06_design_zh_nonverbal",
            "Design ZH + laughter",
            "design",
            "[laughter]今天的发布会到此结束，感谢大家的聆听，祝你晚上愉快。",
            "Chinese",
            instruct="女，青年，中音调",
        ),
        Scenario(
            "07_clone_with_ref_text",
            "Clone ref+text",
            "clone",
            "This cloned sample should preserve the speaking style from the provided reference audio.",
            "English",
            use_ref_audio=True,
            use_ref_text=True,
        ),
        Scenario(
            "08_clone_asr_auto_transcript",
            "Clone ASR",
            "clone",
            "Automatic transcription of the reference should still allow a solid clone.",
            "English",
            use_ref_audio=True,
            use_ref_text=False,
        ),
        Scenario(
            "09_long_chunked_en",
            "Long chunked EN",
            "long",
            long_text,
            "English",
            duration=35.0,
        ),
        Scenario(
            "10_control_speed_slow",
            "Speed 0.85",
            "control",
            "This sentence should be spoken more slowly than the default auto sample.",
            "English",
            speed=0.85,
        ),
        Scenario(
            "11_control_speed_fast",
            "Speed 1.2",
            "control",
            "This sentence should be spoken faster than the default auto sample.",
            "English",
            speed=1.2,
        ),
        Scenario(
            "12_control_fixed_duration",
            "Duration 4.5s",
            "control",
            "Please stretch or compress this line to about four and a half seconds.",
            "English",
            duration=4.5,
        ),
        Scenario(
            "13_control_number_text",
            "Numeric text",
            "control",
            "Please call me at 2345 on March 15, 2026 about invoice 99.",
            "English",
        ),
        Scenario(
            "14_design_whisper",
            "Design whisper",
            "design",
            "Keep this confidential. We will announce the launch next week.",
            "English",
            instruct="female, whisper, young adult",
        ),
        Scenario(
            "15_clone_reuse",
            "Clone (2nd same settings)",
            "clone",
            "Reusing clone settings should sound consistent with the first clone sample.",
            "English",
            use_ref_audio=True,
            use_ref_text=True,
            use_cached_prompt=True,
        ),
    ]


def gen_kwargs() -> dict[str, Any]:
    return {
        "num_step": 32,
        "guidance_scale": 2.0,
        "t_shift": 0.1,
        "denoise": True,
        "position_temperature": 5.0,
        "class_temperature": 0.0,
        "layer_penalty_factor": 5.0,
        "postprocess_output": True,
    }


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_python(
    *,
    model_dir: Path,
    ref_audio: Path,
    ref_text: str,
    device: str,
    seed: int,
    scenarios: list[Scenario],
    repeats: int,
    dtype: str = "f32",
) -> dict[str, Any]:
    import numpy as np
    import torch
    from omnivoice import OmniVoice

    if device.startswith("cuda") and not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available", "device": device, "rows": []}

    dtype_map = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        return {"ok": False, "error": f"unsupported dtype {dtype}", "device": device, "rows": []}
    torch_dtype = dtype_map[dtype]

    print(f"[python] load model on {device} dtype={dtype}", flush=True)
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(str(model_dir), torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    load_s = time.perf_counter() - t0
    param_dtype = next(model.parameters()).dtype
    print(f"[python] loaded in {load_s:.2f}s param_dtype={param_dtype}", flush=True)

    # warmup
    set_seed(seed)
    with torch.inference_mode():
        _ = model.generate(
            text="Warmup.",
            language="English",
            **gen_kwargs(),
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        times: list[float] = []
        audio_sec = None
        err = None
        for rep in range(repeats):
            set_seed(seed + rep)
            call: dict[str, Any] = {
                "text": sc.text,
                "language": sc.language,
                **gen_kwargs(),
            }
            if sc.instruct is not None:
                call["instruct"] = sc.instruct
            if sc.duration is not None:
                call["duration"] = sc.duration
            if sc.speed is not None:
                call["speed"] = sc.speed
            if sc.use_ref_audio:
                if sc.use_cached_prompt:
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=str(ref_audio),
                        ref_text=ref_text if sc.use_ref_text else None,
                    )
                    call["voice_clone_prompt"] = prompt
                else:
                    call["ref_audio"] = str(ref_audio)
                    call["ref_text"] = ref_text if sc.use_ref_text else None
            try:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.inference_mode():
                    audios = model.generate(**call)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t1
                audio = np.asarray(audios[0]).reshape(-1)
                audio_sec = float(audio.shape[0] / getattr(model, "sampling_rate", SAMPLE_RATE))
                times.append(elapsed)
                print(
                    f"[python/{device}] {sc.id} rep{rep+1}/{repeats}: {elapsed:.3f}s "
                    f"(audio {audio_sec:.2f}s, RTF {elapsed/max(audio_sec,1e-9):.3f})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                print(f"[python/{device}] {sc.id} FAILED: {err}", flush=True)
                break

        row: dict[str, Any] = {
            "scenario_id": sc.id,
            "title": sc.title,
            "mode": sc.mode,
            "ok": err is None and bool(times),
            "error": err,
            "wall_s": statistics.median(times) if times else None,
            "wall_s_all": times,
            "audio_s": audio_sec,
            "rtf": (statistics.median(times) / audio_sec) if times and audio_sec else None,
        }
        rows.append(row)

    del model
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()

    return {
        "ok": True,
        "engine": "python-omnivoice",
        "device": device,
        "dtype": dtype,
        "load_s": load_s,
        "rows": rows,
    }


def run_rust(
    *,
    model_dir: Path,
    ref_audio: Path,
    ref_text: str,
    device: str,
    seed: int,
    scenarios: list[Scenario],
    repeats: int,
    release: bool,
    dtype: str = "f16",
) -> dict[str, Any]:
    """Run Rust bench via a one-shot cargo test that times Phase3Pipeline in-process."""
    scenarios_json = ROOT / "artifacts" / "bench_scenarios" / "_scenarios_for_rust.json"
    scenarios_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(model_dir),
        "ref_audio": str(ref_audio),
        "ref_text": ref_text,
        "device": device,
        "dtype": dtype,
        "seed": seed,
        "repeats": repeats,
        "scenarios": [asdict(s) for s in scenarios],
    }
    scenarios_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Write / update the rust bench harness next to tests
    harness = ROOT / "crates" / "omnivoice-infer" / "tests" / "_bench_scenarios.rs"
    harness.write_text(RUST_BENCH_SRC, encoding="utf-8")

    profile = ["--release"] if release else []
    cmd = [
        "cargo",
        "test",
        "-p",
        "omnivoice-infer",
        "--features",
        "cuda",
        *profile,
        "--test",
        "_bench_scenarios",
        "--",
        "--nocapture",
        "--test-threads=1",
    ]
    env = dict(**{k: str(v) for k, v in __import__("os").environ.items()})
    env["OMNIVOICE_BENCH_CONFIG"] = str(scenarios_json)
    print(f"[rust] {' '.join(cmd)} device={device} dtype={dtype}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    marker = "===BENCH_JSON_BEGIN==="
    end = "===BENCH_JSON_END==="
    if marker not in out or end not in out:
        return {
            "ok": False,
            "engine": "rust-omnivoice",
            "device": device,
            "dtype": dtype,
            "error": f"bench output missing JSON (code={proc.returncode})",
            "stdout_tail": out[-4000:],
            "rows": [],
        }
    blob = out.split(marker, 1)[1].split(end, 1)[0].strip()
    data = json.loads(blob)
    data["ok"] = True
    data["engine"] = "rust-omnivoice"
    data["device"] = device
    data["dtype"] = dtype
    data["cargo_returncode"] = proc.returncode
    # print per-row summary
    for row in data.get("rows", []):
        if row.get("ok"):
            print(
                f"[rust/{device}] {row['scenario_id']}: {row['wall_s']:.3f}s "
                f"(audio {row.get('audio_s')}, RTF {row.get('rtf')})",
                flush=True,
            )
        else:
            print(f"[rust/{device}] {row['scenario_id']} FAILED: {row.get('error')}", flush=True)
    return data


RUST_BENCH_SRC = r'''
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
'''


def write_report(out_dir: Path, results: dict[str, Any], scenarios: list[Scenario]) -> None:
    runs = results.get("runs") or {}
    cuda_dtypes = sorted(
        {
            k.rsplit("_", 1)[-1]
            for k in runs
            if k.startswith("python_cuda_") or k.startswith("rust_cuda_")
        }
    )

    lines = [
        "# Benchmark: 15 scenarios × Python/Rust (equal dtype)",
        "",
        f"- created_at: `{results.get('created_at')}`",
        f"- seed: `{results.get('seed')}`",
        f"- repeats (median reported): `{results.get('repeats')}`",
        f"- hardware: `{results.get('hardware', 'NVIDIA GeForce RTX 3060')}`",
        f"- gen: num_step=32, guidance=2.0, t_shift=0.1, position_temperature=5.0, class_temperature=0",
        f"- rule: **same dtype on both engines** (fair comparison)",
        "",
        "## Model load time",
        "",
        "| Engine | Device | Dtype | Load (s) |",
        "|--------|--------|-------|----------|",
    ]
    for key in sorted(runs):
        block = runs.get(key) or {}
        lines.append(
            f"| {block.get('engine', key)} | {block.get('device', '')} | "
            f"{block.get('dtype', '?')} | {block.get('load_s') if block.get('load_s') is not None else '—'} |"
        )

    def wall(key: str, sid: str) -> float | None:
        block = runs.get(key) or {}
        for row in block.get("rows") or []:
            if row.get("scenario_id") == sid:
                return row.get("wall_s")
        return None

    def rtf(key: str, sid: str) -> float | None:
        block = runs.get(key) or {}
        for row in block.get("rows") or []:
            if row.get("scenario_id") == sid:
                return row.get("rtf")
        return None

    def fmt(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{v:.3f}"

    def pct_faster(py: float | None, rs: float | None) -> str:
        if py is None or rs is None or py <= 0:
            return "—"
        return f"{(py - rs) / py * 100.0:+.1f}%"

    for dt in cuda_dtypes:
        pk, rk = f"python_cuda_{dt}", f"rust_cuda_{dt}"
        if pk not in runs and rk not in runs:
            continue
        title = f"CUDA equal dtype: **{dt}**"
        lines += [
            "",
            f"## {title}",
            "",
            "| Scenario | Mode | Python (s) | Rust (s) | Rust/Py | Rust faster |",
            "|----------|------|-----------:|---------:|--------:|------------:|",
        ]
        for sc in scenarios:
            pc, rc = wall(pk, sc.id), wall(rk, sc.id)
            ratio = (rc / pc) if (rc and pc and pc > 0) else None
            lines.append(
                f"| `{sc.id}` | {sc.mode} | {fmt(pc)} | {fmt(rc)} | {fmt(ratio)} | {pct_faster(pc, rc)} |"
            )
        lines += [
            "",
            f"### RTF ({title})",
            "",
            "| Scenario | Python RTF | Rust RTF |",
            "|----------|-----------:|---------:|",
        ]
        for sc in scenarios:
            lines.append(f"| `{sc.id}` | {fmt(rtf(pk, sc.id))} | {fmt(rtf(rk, sc.id))} |")

        py_vals = [wall(pk, sc.id) for sc in scenarios]
        rs_vals = [wall(rk, sc.id) for sc in scenarios]
        py_sum = sum(v for v in py_vals if v is not None)
        rs_sum = sum(v for v in rs_vals if v is not None)
        n_py = sum(1 for v in py_vals if v is not None)
        n_rs = sum(1 for v in rs_vals if v is not None)
        lines += [
            "",
            f"### Totals ({title})",
            "",
            f"- **Python**: {py_sum:.3f}s over {n_py} scenarios",
            f"- **Rust**: {rs_sum:.3f}s over {n_rs} scenarios",
        ]
        if py_sum > 0 and n_py and n_rs:
            lines.append(f"- **Rust vs Python total**: {(py_sum - rs_sum) / py_sum * 100.0:+.1f}%")

    if "python_cpu_f32" in runs or "rust_cpu_f32" in runs:
        lines += [
            "",
            "## CPU (f32 only)",
            "",
            "| Scenario | Mode | Python (s) | Rust (s) | Rust/Py | Rust faster |",
            "|----------|------|-----------:|---------:|--------:|------------:|",
        ]
        for sc in scenarios:
            pc, rc = wall("python_cpu_f32", sc.id), wall("rust_cpu_f32", sc.id)
            ratio = (rc / pc) if (rc and pc and pc > 0) else None
            lines.append(
                f"| `{sc.id}` | {sc.mode} | {fmt(pc)} | {fmt(rc)} | {fmt(ratio)} | {pct_faster(pc, rc)} |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- Wall times exclude model load (load reported separately).",
        "- Warmup: short text generate + clone generate (loads audio tokenizer on Rust).",
        "- Rust uses in-process `Phase3Pipeline` (not CLI cold-start per scenario).",
        "- Ratio < 1 means Rust is faster; percent is (Python − Rust) / Python.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    p.add_argument("--ref-text-file", default=str(DEFAULT_REF_TEXT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--repeats", type=int, default=1, help="runs per scenario; report median")
    p.add_argument("--skip-python", action="store_true")
    p.add_argument("--skip-rust", action="store_true")
    p.add_argument("--rust-release", action="store_true", help="cargo test --release for Rust")
    p.add_argument("--devices", default="cuda", help="comma list: cuda,cpu")
    p.add_argument(
        "--dtypes",
        default="f16",
        help="comma list for equal-condition CUDA pairs: f32,f16,bf16 (CPU always f32)",
    )
    p.add_argument(
        "--hardware",
        default="NVIDIA GeForce RTX 3060",
        help="label written into the report",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model = Path(args.model).resolve()
    ref_audio = Path(args.ref_audio).resolve()
    ref_text_file = Path(args.ref_text_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_text = ref_text_file.read_text(encoding="utf-8").strip()
    scenarios = build_scenarios()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    dtypes = [d.strip().lower() for d in args.dtypes.split(",") if d.strip()]

    results: dict[str, Any] = {
        "created_at": utc_now(),
        "seed": args.seed,
        "repeats": args.repeats,
        "model": str(model),
        "hardware": args.hardware,
        "dtypes": dtypes,
        "runs": {},
    }

    if not args.skip_python:
        for dev in devices:
            is_cuda = dev.startswith("cuda")
            for dt in dtypes if is_cuda else ["f32"]:
                key = f"python_{'cuda' if is_cuda else 'cpu'}_{dt}"
                results["runs"][key] = run_python(
                    model_dir=model,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    device="cuda:0" if is_cuda else "cpu",
                    seed=args.seed,
                    scenarios=scenarios,
                    repeats=args.repeats,
                    dtype=dt,
                )

    if not args.skip_rust:
        for dev in devices:
            is_cuda = dev.startswith("cuda")
            for dt in dtypes if is_cuda else ["f32"]:
                key = f"rust_{'cuda' if is_cuda else 'cpu'}_{dt}"
                results["runs"][key] = run_rust(
                    model_dir=model,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    device="cuda:0" if is_cuda else "cpu",
                    seed=args.seed,
                    scenarios=scenarios,
                    repeats=args.repeats,
                    release=args.rust_release,
                    dtype=dt,
                )

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(out_dir, results, scenarios)
    print(f"wrote {out_dir / 'README.md'}", flush=True)
    print(f"wrote {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
