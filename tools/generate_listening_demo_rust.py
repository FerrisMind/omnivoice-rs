#!/usr/bin/env python3
"""Generate manual listening demos from the **Rust** omnivoice-cli (not Python).

Runs the same usage scenarios on CUDA first, then CPU, writing WAV + README.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "v2" / "model"
DEFAULT_REF_AUDIO = ROOT / "artifacts" / "v2" / "ref.wav"
DEFAULT_REF_TEXT = ROOT / "artifacts" / "v2" / "ref_text.txt"
DEFAULT_OUT = ROOT / "artifacts" / "listening_demo_rust"
DEFAULT_CLI = ROOT / "target" / "debug" / "omnivoice-cli.exe"
DEFAULT_SEED = 1234


@dataclass
class Scenario:
    id: str
    title: str
    mode: str
    description: str
    text: str
    language: str | None = None
    instruct: str | None = None
    use_ref_audio: bool = False
    use_ref_text: bool = False
    duration: float | None = None
    speed: float | None = None
    listen_for: str = ""
    extra_cli: list[str] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def build_scenarios(ref_text: str) -> list[Scenario]:
    long_text = (
        "OmniVoice can synthesize long-form speech while keeping memory usage stable. "
        "In this listening check, the text is intentionally longer than a typical demo "
        "sentence so the pipeline is forced through its chunking path. Each section "
        "should stay intelligible, maintain similar pacing, and join smoothly with the "
        "following section. We repeat the core idea so the model has enough text to split. "
        "OmniVoice can synthesize long-form speech while keeping memory usage stable."
    )
    return [
        Scenario(
            id="01_auto_en_short",
            title="Auto voice — English short",
            mode="auto",
            description="Rust infer: text + language only (no ref, no instruct).",
            text="OmniVoice creates clear speech from text with minimal setup.",
            language="English",
            listen_for="Natural English auto voice from Rust.",
        ),
        Scenario(
            id="02_auto_zh_short",
            title="Auto voice — Chinese short",
            mode="auto",
            description="Rust infer: Chinese language tag.",
            text="欢迎使用 OmniVoice。这是一段中文自动音色的试听样本。",
            language="Chinese",
            listen_for="Chinese auto voice.",
        ),
        Scenario(
            id="03_auto_lang_agnostic",
            title="Auto voice — no language flag",
            mode="auto",
            description="Rust infer without --language.",
            text="This sample leaves language unspecified so the model chooses freely.",
            language=None,
            listen_for="Still intelligible without explicit language.",
        ),
        Scenario(
            id="04_design_en_british_female",
            title="Voice design — British female",
            mode="design",
            description="Rust infer with --instruct only.",
            text=(
                "Good afternoon. This reference should sound calm, precise, "
                "and suitable for a polished product demo."
            ),
            language="English",
            instruct="female, low pitch, british accent",
            listen_for="Female British product-demo tone.",
        ),
        Scenario(
            id="05_design_en_male_american",
            title="Voice design — male American",
            mode="design",
            description="Different supported instruct tags.",
            text="Hey there! Ready to ship another release? Let's make this one count.",
            language="English",
            instruct="male, high pitch, young adult, american accent",
            listen_for="Different from 04 — younger male American.",
        ),
        Scenario(
            id="06_design_zh_nonverbal",
            title="Voice design — Chinese + [laughter]",
            mode="design",
            description="Chinese instruct (full-width commas) + nonverbal tag.",
            text="[laughter]今天的发布会到此结束，感谢大家的聆听，祝你晚上愉快。",
            language="Chinese",
            instruct="女，青年，中音调",
            listen_for="Laughter then Chinese line.",
        ),
        Scenario(
            id="07_clone_with_ref_text",
            title="Voice clone — ref audio + ref text",
            mode="clone",
            description="Rust infer --ref-audio + --ref-text.",
            text=(
                "This cloned sample should preserve the speaking style from "
                "the provided reference audio."
            ),
            language="English",
            use_ref_audio=True,
            use_ref_text=True,
            listen_for=f"Should match ref.wav speaker (ref text starts: “{ref_text[:60]}…”).",
        ),
        Scenario(
            id="08_clone_asr_auto_transcript",
            title="Voice clone — ref audio only (ASR)",
            mode="clone",
            description="Rust infer --ref-audio without --ref-text (ASR path if enabled).",
            text="Automatic transcription of the reference should still allow a solid clone.",
            language="English",
            use_ref_audio=True,
            use_ref_text=False,
            listen_for="Same speaker as 07 if ASR works in Rust CLI.",
        ),
        Scenario(
            id="09_long_chunked_en",
            title="Long-form chunked — English",
            mode="long",
            description="Long text + --duration to force long/chunked path.",
            text=long_text,
            language="English",
            duration=35.0,
            listen_for="Long continuous speech, smooth joins.",
        ),
        Scenario(
            id="10_control_speed_slow",
            title="Control — speed 0.85",
            mode="control",
            description="--speed 0.85",
            text="This sentence should be spoken more slowly than the default auto sample.",
            language="English",
            speed=0.85,
            listen_for="Slower than 01.",
        ),
        Scenario(
            id="11_control_speed_fast",
            title="Control — speed 1.2",
            mode="control",
            description="--speed 1.2",
            text="This sentence should be spoken faster than the default auto sample.",
            language="English",
            speed=1.2,
            listen_for="Faster than 01.",
        ),
        Scenario(
            id="12_control_fixed_duration",
            title="Control — fixed duration 4.5s",
            mode="control",
            description="--duration 4.5",
            text="Please stretch or compress this line to about four and a half seconds.",
            language="English",
            duration=4.5,
            listen_for="Duration near 4.5s in player.",
        ),
        Scenario(
            id="13_control_number_text",
            title="Control — numeric text",
            mode="control",
            description="Digits/dates without optional TN package.",
            text="Please call me at 2345 on March 15, 2026 about invoice 99.",
            language="English",
            listen_for="How Rust reads digits/dates.",
        ),
        Scenario(
            id="14_design_whisper",
            title="Voice design — whisper",
            mode="design",
            description="Supported instruct: female, whisper, young adult.",
            text="Keep this confidential. We will announce the launch next week.",
            language="English",
            instruct="female, whisper, young adult",
            listen_for="Whispered / soft delivery.",
        ),
        Scenario(
            id="15_clone_same_as_07_seeded",
            title="Voice clone — second seeded clone",
            mode="clone",
            description=(
                "Same clone inputs as 07 with same seed (Rust CLI has no separate "
                "create_voice_clone_prompt subcommand exposed the same way; this "
                "checks clone stability)."
            ),
            text="Reusing clone settings should sound consistent with the first clone sample.",
            language="English",
            use_ref_audio=True,
            use_ref_text=True,
            listen_for="Similar speaker to 07.",
        ),
    ]


def run_infer(
    *,
    cli: Path,
    model: Path,
    scenario: Scenario,
    ref_audio: Path,
    ref_text: str,
    device: str,
    dtype: str,
    seed: int,
    out_wav: Path,
) -> dict[str, Any]:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli),
        "infer",
        "--model",
        str(model),
        "--text",
        scenario.text,
        "--output",
        str(out_wav),
        "--device",
        device,
        "--dtype",
        dtype,
        "--seed",
        str(seed),
        "--num-step",
        "32",
        "--guidance-scale",
        "2.0",
        "--t-shift",
        "0.1",
        "--denoise",
        "true",
        "--postprocess-output",
        "true",
    ]
    if scenario.language is not None:
        cmd.extend(["--language", scenario.language])
    if scenario.instruct is not None:
        cmd.extend(["--instruct", scenario.instruct])
    if scenario.use_ref_audio:
        cmd.extend(["--ref-audio", str(ref_audio)])
        if scenario.use_ref_text:
            cmd.extend(["--ref-text", ref_text])
    if scenario.duration is not None:
        cmd.extend(["--duration", str(scenario.duration)])
    if scenario.speed is not None:
        cmd.extend(["--speed", str(scenario.speed)])
    cmd.extend(scenario.extra_cli)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    meta: dict[str, Any] = {
        "scenario_id": scenario.id,
        "engine": "omnivoice-rs / omnivoice-cli",
        "device": device,
        "dtype": dtype,
        "seed": seed,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
    if proc.returncode != 0 or not out_wav.is_file():
        meta["ok"] = False
        meta["error"] = (proc.stderr or proc.stdout or "infer failed").strip()[-500:]
        return meta
    meta["ok"] = True
    meta["wav"] = str(out_wav)
    try:
        meta["duration_sec"] = wav_duration_sec(out_wav)
    except Exception as exc:  # noqa: BLE001
        meta["duration_sec"] = None
        meta["duration_error"] = str(exc)
    return meta


def write_readme(
    out_dir: Path,
    scenarios: list[Scenario],
    *,
    model: Path,
    ref_audio: Path,
    devices: list[str],
    results: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Listening demo — **Rust omnivoice-rs**",
        "",
        "> Это **наш** проект (`omnivoice-cli`), **не** Python OmniVoice reference.",
        ">",
        "> Python-эталон, если нужен, лежит отдельно в `artifacts/listening_demo/`.",
        "",
        "Сгенерировано: `tools/generate_listening_demo_rust.py`",
        "",
        "## Как слушать",
        "",
        "1. Открой `cuda/` (GPU), затем `cpu/`.",
        "2. Одинаковые имена файлов = один сценарий на двух устройствах.",
        "3. Рядом `.json` с командой CLI и статусом.",
        "4. Сравнивай `cuda/NN_....wav` ↔ `cpu/NN_....wav`.",
        "",
        "## Окружение",
        "",
        f"- CLI: `omnivoice-cli infer`",
        f"- model: `{model}`",
        f"- ref audio: `{ref_audio}`",
        f"- devices: `{', '.join(devices)}`",
        f"- seed: `1234`",
        f"- created_at: `{utc_now()}`",
        "",
        "## Карта сценариев → файлы",
        "",
        "| # | File | Mode | Сценарий | На что слушать |",
        "|---|------|------|----------|----------------|",
    ]
    for s in scenarios:
        lines.append(
            f"| {s.id[:2]} | `{{device}}/{s.id}.wav` | `{s.mode}` | {s.title} | {s.listen_for} |"
        )
    lines.append("")
    lines.append("`{device}` = `cuda` или `cpu`.")
    lines.append("")
    lines.append("## Подробно")
    lines.append("")
    for s in scenarios:
        lines.append(f"### `{s.id}.wav` — {s.title}")
        lines.append("")
        lines.append(f"- **mode:** `{s.mode}`")
        lines.append(f"- **description:** {s.description}")
        lines.append(f"- **language:** `{s.language}`")
        if s.instruct:
            lines.append(f"- **instruct:** `{s.instruct}`")
        if s.use_ref_audio:
            lines.append(
                f"- **clone:** ref_audio=yes, ref_text={'yes' if s.use_ref_text else 'ASR/none'}"
            )
        if s.duration is not None:
            lines.append(f"- **duration:** {s.duration}")
        if s.speed is not None:
            lines.append(f"- **speed:** {s.speed}")
        lines.append(f"- **text:** {s.text}")
        lines.append(f"- **listen for:** {s.listen_for}")
        for device in devices:
            rows = [r for r in results.get(device, []) if r.get("scenario_id") == s.id]
            if not rows:
                lines.append(f"- **{device}:** missing")
            else:
                r = rows[0]
                if r.get("ok"):
                    dur = r.get("duration_sec")
                    dur_s = f"{dur:.2f}s" if isinstance(dur, (int, float)) else "?"
                    lines.append(f"- **{device}:** ok, duration≈{dur_s}")
                else:
                    lines.append(f"- **{device}:** FAILED — {r.get('error', 'unknown')[:200]}")
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli", default=str(DEFAULT_CLI))
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    p.add_argument("--ref-text-file", default=str(DEFAULT_REF_TEXT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--devices", default="cuda:0,cpu")
    p.add_argument(
        "--dtype-cuda",
        default="f32",
        help="dtype for CUDA pass (default f32 for closer parity)",
    )
    p.add_argument("--dtype-cpu", default="f32")
    p.add_argument("--only", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cli = Path(args.cli).resolve()
    model = Path(args.model).resolve()
    ref_audio = Path(args.ref_audio).resolve()
    ref_text_file = Path(args.ref_text_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cli.is_file():
        print(f"missing CLI binary: {cli}", file=sys.stderr)
        print("Build first: cargo build -p omnivoice-cli --features cuda", file=sys.stderr)
        return 1
    for path, label in ((model, "model"), (ref_audio, "ref audio"), (ref_text_file, "ref text")):
        if not path.exists():
            print(f"missing {label}: {path}", file=sys.stderr)
            return 1

    ref_text = ref_text_file.read_text(encoding="utf-8").strip()
    scenarios = build_scenarios(ref_text)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        scenarios = [s for s in scenarios if s.id in only]

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    results: dict[str, list[dict[str, Any]]] = {}
    devices_done: list[str] = []

    (out_dir / "scenarios.json").write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "engine": "omnivoice-rs",
                "cli": str(cli),
                "model": str(model),
                "scenarios": [asdict(s) for s in scenarios],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for device in devices:
        key = device.replace(":", "_")
        dtype = args.dtype_cpu if device == "cpu" else args.dtype_cuda
        device_dir = out_dir / key
        device_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        print(f"=== device={device} dtype={dtype} ===", flush=True)
        for scenario in scenarios:
            wav = device_dir / f"{scenario.id}.wav"
            meta_path = device_dir / f"{scenario.id}.json"
            print(f"[{device}] {scenario.id} ...", flush=True)
            try:
                meta = run_infer(
                    cli=cli,
                    model=model,
                    scenario=scenario,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    device=device,
                    dtype=dtype,
                    seed=args.seed,
                    out_wav=wav,
                )
            except Exception as exc:  # noqa: BLE001
                meta = {
                    "scenario_id": scenario.id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            rows.append(meta)
            if meta.get("ok"):
                print(
                    f"[{device}] {scenario.id} ok  ~{meta.get('duration_sec', '?')}s",
                    flush=True,
                )
            else:
                print(f"[{device}] {scenario.id} FAILED: {meta.get('error')}", flush=True)
        results[key] = rows
        devices_done.append(key)

    (out_dir / "results.json").write_text(
        json.dumps(
            {"created_at": utc_now(), "devices": devices_done, "results": results},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_readme(
        out_dir,
        scenarios,
        model=model,
        ref_audio=ref_audio,
        devices=devices_done,
        results=results,
    )
    print(f"Rust listening demo written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
