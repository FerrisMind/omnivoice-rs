#!/usr/bin/env python3
"""Generate manual listening demos from the official Python OmniVoice reference.

Runs every supported usage scenario first on CUDA, then on CPU, writing WAV
files plus a human-readable README so you can A/B listen without reading code.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "v2" / "model"
DEFAULT_REF_AUDIO = ROOT / "artifacts" / "v2" / "ref.wav"
DEFAULT_REF_TEXT = ROOT / "artifacts" / "v2" / "ref_text.txt"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "listening_demo"
DEFAULT_SEED = 1234
SAMPLE_RATE = 24_000


@dataclass
class Scenario:
    """One listen-able usage scenario."""

    id: str
    title: str
    mode: str  # auto | design | clone | long | control
    description: str
    text: str
    language: str | None = None
    instruct: str | None = None
    use_ref_audio: bool = False
    use_ref_text: bool = False  # if use_ref_audio and False → ASR path
    duration: float | None = None
    speed: float | None = None
    normalize_text: bool = False
    generation: dict[str, Any] = field(default_factory=dict)
    listen_for: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_ref_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scenarios(ref_text: str) -> list[Scenario]:
    """All user-facing OmniVoice modes we want to hear manually."""
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
            description=(
                "Neither reference audio nor style instruct. Model picks a voice. "
                "Language is explicitly English."
            ),
            text="OmniVoice creates clear speech from text with minimal setup.",
            language="English",
            listen_for="Natural English voice, clear phrasing, no clone of ref.wav.",
        ),
        Scenario(
            id="02_auto_zh_short",
            title="Auto voice — Chinese short",
            mode="auto",
            description="Auto voice with Chinese language tag.",
            text="欢迎使用 OmniVoice。这是一段中文自动音色的试听样本。",
            language="Chinese",
            listen_for="Chinese auto voice, not English clone.",
        ),
        Scenario(
            id="03_auto_lang_agnostic",
            title="Auto voice — language-agnostic",
            mode="auto",
            description="language=None (language-agnostic mode).",
            text="This sample leaves language unspecified so the model chooses freely.",
            language=None,
            listen_for="Should still be intelligible English-ish speech without explicit lang.",
        ),
        Scenario(
            id="04_design_en_british_female",
            title="Voice design — British female",
            mode="design",
            description="Style control via instruct text only (no reference audio).",
            text=(
                "Good afternoon. This reference should sound calm, precise, "
                "and suitable for a polished product demo."
            ),
            language="English",
            instruct="female, low pitch, british accent",
            listen_for="Female British accent, calm product-demo tone.",
        ),
        Scenario(
            id="05_design_en_male_american",
            title="Voice design — male American high pitch",
            mode="design",
            description=(
                "Different instruct attributes using only OmniVoice-supported "
                "English tags (see voice-design docs)."
            ),
            text="Hey there! Ready to ship another release? Let's make this one count.",
            language="English",
            instruct="male, high pitch, young adult, american accent",
            listen_for="Clearly different from 04 — younger male American voice.",
        ),
        Scenario(
            id="06_design_zh_nonverbal",
            title="Voice design — Chinese + nonverbal tag",
            mode="design",
            description=(
                "Chinese text with [laughter] control tag. Instruct uses "
                "full-width Chinese attribute list."
            ),
            text="[laughter]今天的发布会到此结束，感谢大家的聆听，祝你晚上愉快。",
            language="Chinese",
            instruct="女，青年，中音调",
            listen_for="Brief laughter then Chinese closing line.",
        ),
        Scenario(
            id="07_clone_with_ref_text",
            title="Voice clone — ref audio + ref text",
            mode="clone",
            description=(
                "Classic zero-shot clone: provide ref.wav and its transcript. "
                "Target text is different from the reference transcript."
            ),
            text=(
                "This cloned sample should preserve the speaking style from "
                "the provided reference audio."
            ),
            language="English",
            use_ref_audio=True,
            use_ref_text=True,
            listen_for=f"Should match speaker/style of ref.wav (ref text: “{ref_text[:80]}…”).",
        ),
        Scenario(
            id="08_clone_asr_auto_transcript",
            title="Voice clone — ref audio only (ASR transcript)",
            mode="clone",
            description=(
                "Clone with ref_audio but without ref_text. OmniVoice ASR "
                "auto-transcribes the reference clip."
            ),
            text="Automatic transcription of the reference should still allow a solid clone.",
            language="English",
            use_ref_audio=True,
            use_ref_text=False,
            listen_for="Same speaker as 07 if ASR succeeds; may fail if ASR deps missing.",
        ),
        Scenario(
            id="09_long_chunked_en",
            title="Long-form chunked — English",
            mode="long",
            description=(
                "Long text forces chunked generation (audio_chunk_threshold / duration). "
                "Checks stitching and pacing across chunks."
            ),
            text=long_text,
            language="English",
            duration=35.0,
            listen_for="Long continuous speech, smooth joins, no harsh cuts between chunks.",
        ),
        Scenario(
            id="10_control_speed_slow",
            title="Control — slow speed (0.85×)",
            mode="control",
            description="Auto English with speed=0.85 (slower speech).",
            text="This sentence should be spoken more slowly than the default auto sample.",
            language="English",
            speed=0.85,
            listen_for="Noticeably slower than 01_auto_en_short.",
        ),
        Scenario(
            id="11_control_speed_fast",
            title="Control — fast speed (1.2×)",
            mode="control",
            description="Auto English with speed=1.2 (faster speech).",
            text="This sentence should be spoken faster than the default auto sample.",
            language="English",
            speed=1.2,
            listen_for="Noticeably faster than 01_auto_en_short.",
        ),
        Scenario(
            id="12_control_fixed_duration",
            title="Control — fixed duration (4.5 s)",
            mode="control",
            description="duration=4.5 forces approximate output length (overrides speed).",
            text="Please stretch or compress this line to about four and a half seconds.",
            language="English",
            duration=4.5,
            listen_for="Clip length near 4.5s (check WAV duration in player).",
        ),
        Scenario(
            id="13_control_number_text",
            title="Control — numeric text (no external TN)",
            mode="control",
            description=(
                "Speaks a line with digits without WeTextProcessing "
                "(normalize_text=False). Shows default number reading."
            ),
            text="Please call me at 2345 on March 15, 2026 about invoice 99.",
            language="English",
            normalize_text=False,
            listen_for="How digits/dates are spoken without optional text-normalization deps.",
        ),
        Scenario(
            id="14_design_whisper",
            title="Voice design — whisper",
            mode="design",
            description="Instruct uses only supported tags: female + whisper.",
            text="Keep this confidential. We will announce the launch next week.",
            language="English",
            instruct="female, whisper, young adult",
            listen_for="Quiet / whispered quality.",
        ),
        Scenario(
            id="15_clone_then_reuse_prompt",
            title="Voice clone — create_voice_clone_prompt reuse",
            mode="clone",
            description=(
                "Builds VoiceClonePrompt once via create_voice_clone_prompt, "
                "then generates with voice_clone_prompt=... (API reuse path)."
            ),
            text="Reusing a cached clone prompt should sound consistent with direct clone.",
            language="English",
            use_ref_audio=True,
            use_ref_text=True,
            generation={"_use_cached_prompt": True},
            listen_for="Similar speaker to 07_clone_with_ref_text.",
        ),
    ]


def default_gen_kwargs() -> dict[str, Any]:
    return {
        "num_step": 32,
        "guidance_scale": 2.0,
        "t_shift": 0.1,
        "denoise": True,
        "position_temperature": 5.0,
        "class_temperature": 0.0,
        "layer_penalty_factor": 5.0,
        "preprocess_prompt": True,
        "postprocess_output": True,
    }


def load_model(model_dir: Path, device: str):
    from omnivoice import OmniVoice

    print(f"loading model from {model_dir} on {device}", flush=True)
    model = OmniVoice.from_pretrained(str(model_dir))
    model = model.to(device)
    model.eval()
    return model


def generate_one(
    model,
    scenario: Scenario,
    *,
    ref_audio: Path,
    ref_text: str,
    device: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    kwargs = default_gen_kwargs()
    extra = dict(scenario.generation)
    use_cached_prompt = bool(extra.pop("_use_cached_prompt", False))
    kwargs.update(extra)

    call: dict[str, Any] = {
        "text": scenario.text,
        "language": scenario.language,
        "normalize_text": scenario.normalize_text,
    }
    if scenario.instruct is not None:
        call["instruct"] = scenario.instruct
    if scenario.duration is not None:
        call["duration"] = scenario.duration
    if scenario.speed is not None:
        call["speed"] = scenario.speed

    if scenario.use_ref_audio:
        if use_cached_prompt:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text if scenario.use_ref_text else None,
            )
            call["voice_clone_prompt"] = prompt
        else:
            call["ref_audio"] = str(ref_audio)
            if scenario.use_ref_text:
                call["ref_text"] = ref_text
            else:
                call["ref_text"] = None

    # Keep generation kwargs explicit for reproducibility notes.
    call.update(kwargs)

    with torch.inference_mode():
        audios = model.generate(**call)
    audio = np.asarray(audios[0], dtype=np.float32).reshape(-1)
    meta = {
        "scenario_id": scenario.id,
        "device": device,
        "seed": seed,
        "sample_rate": int(getattr(model, "sampling_rate", SAMPLE_RATE)),
        "num_samples": int(audio.shape[0]),
        "duration_sec": float(audio.shape[0] / getattr(model, "sampling_rate", SAMPLE_RATE)),
        "peak_abs": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "call": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in call.items()
            if k != "voice_clone_prompt" and k != "ref_audio"
        },
        "used_ref_audio": scenario.use_ref_audio,
        "used_cached_prompt": use_cached_prompt,
    }
    if scenario.use_ref_audio:
        meta["call"]["ref_audio"] = str(ref_audio)
    return audio, meta


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def write_readme(
    out_dir: Path,
    scenarios: list[Scenario],
    *,
    model_dir: Path,
    ref_audio: Path,
    devices_done: list[str],
    results: dict[str, list[dict[str, Any]]],
) -> None:
    lines: list[str] = []
    lines.append("# OmniVoice listening demo")
    lines.append("")
    lines.append("Ручная прослушка эталона **Python OmniVoice** (upstream reference).")
    lines.append("")
    lines.append("Сгенерировано скриптом `tools/python_reference/generate_listening_demo.py`.")
    lines.append("")
    lines.append("## Как слушать")
    lines.append("")
    lines.append("1. Сначала папка `cuda/` (если есть), потом `cpu/`.")
    lines.append("2. Имена файлов = `NN_scenario_id.wav` — одинаковые на обоих устройствах.")
    lines.append("3. Рядом с каждым WAV лежит `NN_scenario_id.json` с параметрами вызова.")
    lines.append("4. Сравнивайте **один и тот же номер** между CUDA и CPU.")
    lines.append("")
    lines.append("## Окружение")
    lines.append("")
    lines.append(f"- model: `{model_dir}`")
    lines.append(f"- ref audio (clone): `{ref_audio}`")
    lines.append(f"- devices run: `{', '.join(devices_done)}`")
    lines.append(f"- created_at: `{utc_now()}`")
    lines.append("- seed: `1234`")
    lines.append("- sampling rate: `24000 Hz`, PCM_16 WAV")
    lines.append("")
    lines.append("## Карта сценариев → файлы")
    lines.append("")
    lines.append("| # | File | Mode | What it tests | What to listen for |")
    lines.append("|---|------|------|---------------|--------------------|")
    for s in scenarios:
        wav = f"`{{device}}/{s.id}.wav`"
        lines.append(
            f"| {s.id.split('_', 1)[0]} | {wav} | `{s.mode}` | {s.title} | {s.listen_for} |"
        )
    lines.append("")
    lines.append("Подставьте `device` = `cuda` или `cpu`.")
    lines.append("")
    lines.append("## Подробно по сценариям")
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
                f"- **clone:** ref_audio=yes, ref_text={'yes' if s.use_ref_text else 'ASR/auto'}"
            )
        if s.duration is not None:
            lines.append(f"- **duration:** `{s.duration}` s")
        if s.speed is not None:
            lines.append(f"- **speed:** `{s.speed}`")
        if s.normalize_text:
            lines.append("- **normalize_text:** `true`")
        lines.append(f"- **text:** {s.text}")
        lines.append(f"- **listen for:** {s.listen_for}")
        lines.append("")
        for device in devices_done:
            entries = [r for r in results.get(device, []) if r.get("scenario_id") == s.id]
            if not entries:
                lines.append(f"- **{device}:** missing / failed (see `results.json`)")
            else:
                r = entries[0]
                if r.get("ok"):
                    lines.append(
                        f"- **{device}:** ok, duration≈{r['duration_sec']:.2f}s, peak={r['peak_abs']:.3f}"
                    )
                else:
                    lines.append(f"- **{device}:** FAILED — {r.get('error', 'unknown')}")
        lines.append("")
    lines.append("## Режимы OmniVoice (кратко)")
    lines.append("")
    lines.append("| Mode | API inputs |")
    lines.append("|------|------------|")
    lines.append("| **auto** | `text` (+ optional `language`) |")
    lines.append("| **design** | `text` + `instruct` |")
    lines.append("| **clone** | `text` + `ref_audio` + `ref_text` (or ASR) / `voice_clone_prompt` |")
    lines.append("| **long** | long `text` / `duration` → chunked generation |")
    lines.append("| **control** | `speed`, `duration`, `normalize_text`, tags like `[laughter]` |")
    lines.append("")
    lines.append("Источник API: upstream `omnivoice.models.omnivoice.OmniVoice.generate`.")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    p.add_argument("--ref-text-file", default=str(DEFAULT_REF_TEXT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--devices",
        default="cuda,cpu",
        help="comma list, order matters (default: cuda then cpu)",
    )
    p.add_argument(
        "--only",
        default="",
        help="optional comma-separated scenario ids to run (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    ref_audio = Path(args.ref_audio).resolve()
    ref_text_file = Path(args.ref_text_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, label in (
        (model_dir, "model"),
        (ref_audio, "ref audio"),
        (ref_text_file, "ref text"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    ref_text = read_ref_text(ref_text_file)
    scenarios = build_scenarios(ref_text)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        scenarios = [s for s in scenarios if s.id in only]

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    results: dict[str, list[dict[str, Any]]] = {}
    devices_done: list[str] = []

    catalog = {
        "created_at": utc_now(),
        "model_dir": str(model_dir),
        "ref_audio": str(ref_audio),
        "ref_text": ref_text,
        "seed": args.seed,
        "scenarios": [asdict(s) for s in scenarios],
    }
    (out_dir / "scenarios.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"skip {device}: CUDA not available", flush=True)
            continue
        if device == "cpu":
            pass
        elif device.startswith("cuda"):
            pass
        else:
            print(f"skip unknown device {device}", flush=True)
            continue

        device_dir = out_dir / device.replace(":", "_")
        device_dir.mkdir(parents=True, exist_ok=True)
        model = load_model(model_dir, device)
        device_results: list[dict[str, Any]] = []

        for scenario in scenarios:
            wav_path = device_dir / f"{scenario.id}.wav"
            json_path = device_dir / f"{scenario.id}.json"
            print(f"[{device}] {scenario.id} ...", flush=True)
            try:
                audio, meta = generate_one(
                    model,
                    scenario,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    device=device,
                    seed=args.seed,
                )
                write_wav(wav_path, audio, meta["sample_rate"])
                meta["ok"] = True
                meta["wav"] = str(wav_path.relative_to(out_dir)).replace("\\", "/")
                json_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                device_results.append(meta)
                print(
                    f"[{device}] {scenario.id} ok  {meta['duration_sec']:.2f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - keep demo resilient
                err = {
                    "scenario_id": scenario.id,
                    "device": device,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                device_results.append(err)
                json_path.write_text(
                    json.dumps(err, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                print(f"[{device}] {scenario.id} FAILED: {exc}", flush=True)

        results[device.replace(":", "_")] = device_results
        devices_done.append(device.replace(":", "_"))
        # free GPU memory between device passes
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        model_dir=model_dir,
        ref_audio=ref_audio,
        devices_done=devices_done,
        results=results,
    )
    print(f"listening demo written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
