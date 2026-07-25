#!/usr/bin/env python3
"""Freeze CPU baselines that mirror the GPU v2 set (same cases / captures).

Directory layout parallels the GPU pack under ``artifacts/v2/artifacts``:

- ``python_reference_cpu_strict`` ↔ ``python_reference``
- ``python_reference_stage0_deterministic_cpu_strict`` ↔ ``python_reference_stage0_deterministic``
- ``python_reference_stage0_cpu_debug`` ↔ ``python_reference_stage0_cuda_debug``
- ``python_reference_stage7_cpu_f32_dense`` ↔ ``python_reference_stage7_cuda_f32_dense``

All inference runs on CPU float32.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPORT_REFERENCE = ROOT / "tools" / "python_reference" / "export_reference.py"
VALIDATE_REFERENCE = ROOT / "tools" / "python_reference" / "validate_reference.py"
DEFAULT_V2_ROOT = ROOT / "artifacts" / "v2"
DEFAULT_MODEL_DIR = DEFAULT_V2_ROOT / "model"
DEFAULT_REF_AUDIO = DEFAULT_V2_ROOT / "ref.wav"
DEFAULT_REF_TEXT = DEFAULT_V2_ROOT / "ref_text.txt"
CPU_DEVICE = "cpu"

# Keep in lockstep with freeze_v2_gpu.py
PRODUCT_CASE_IDS = [
    "auto_en_short",
    "design_en_british",
    "design_zh_control",
    "clone_user_ref",
    "auto_long_chunked",
    "debug_auto_en_short",
    "debug_clone_user_ref",
]
PRODUCT_DEBUG_CASE_IDS = ["debug_auto_en_short", "debug_clone_user_ref"]
STAGE0_CASE_IDS = [
    "det_auto_en_short",
    "det_design_en_british",
    "det_clone_user_ref",
    "det_auto_long_chunked",
]
STAGE0_DEBUG_CASE_IDS = ["det_debug_auto_en_short", "det_debug_clone_user_ref"]
DENSE_CASE_IDS = STAGE0_CASE_IDS
CAPTURE_STEPS = ",".join(str(step) for step in range(32))
DENSE_CAPTURE_LAYERS = ",".join([*(str(layer) for layer in range(28)), "final"])
DEBUG_CAPTURE_STEPS = "0,15,31"
DEBUG_CAPTURE_LAYERS = "0,13,27,final"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_export(
    *,
    model_dir: Path,
    ref_audio: Path,
    ref_text_file: Path,
    out_dir: Path,
    case_ids: list[str],
    debug_case_ids: list[str] | None = None,
    capture_steps: str | None = None,
    capture_layers: str | None = None,
    capture_stage1_debug: bool = False,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    command = [
        sys.executable,
        str(EXPORT_REFERENCE),
        "--model-dir",
        str(model_dir),
        "--ref-audio",
        str(ref_audio),
        "--ref-text-file",
        str(ref_text_file),
        "--out-dir",
        str(out_dir),
        "--device",
        CPU_DEVICE,
        "--dtype",
        "float32",
        "--seed",
        "1234",
        "--case-ids",
        ",".join(case_ids),
    ]
    if debug_case_ids:
        command.extend(["--debug-case-ids", ",".join(debug_case_ids)])
        command.extend(["--debug-device", CPU_DEVICE])
        command.extend(["--debug-dtype", "float32"])
    if capture_steps:
        command.extend(["--capture-steps", capture_steps])
    if capture_layers:
        command.extend(["--capture-layers", capture_layers])
    if capture_stage1_debug:
        command.append("--capture-stage1-debug")
    print(f"exporting {out_dir.name}: cases={case_ids}", flush=True)
    subprocess.run(command, check=True)


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_cpu_index(
    v2_root: Path, model_dir: Path, ref_audio: Path, ref_text_file: Path
) -> dict[str, Any]:
    def manifest_for(name: str) -> str:
        return f"{name}/manifest.json"

    return {
        "version": "v2-cpu",
        "created_at": utc_now(),
        "root": str(v2_root),
        "source_root": str(ROOT),
        "reference_commit": git_commit(ROOT / ".." / "omnivoice_refs" / "OmniVoice"),
        "model_dir": str(model_dir),
        "ref_audio": str(ref_audio),
        "ref_text_file": str(ref_text_file),
        "gpu_only": False,
        "seed": 1234,
        "mirrors_gpu_case_set": True,
        "baselines": {
            "cpu_product_f32": {
                "manifest": manifest_for("python_reference_cpu_strict"),
                "device": CPU_DEVICE,
                "dtype": "float32",
                "case_ids": PRODUCT_CASE_IDS,
                "mirrors": "gpu_product_f16",
            },
            "cpu_stage0_f32": {
                "manifest": manifest_for("python_reference_stage0_deterministic_cpu_strict"),
                "device": CPU_DEVICE,
                "dtype": "float32",
                "case_ids": STAGE0_CASE_IDS,
                "mirrors": "gpu_stage0_f16",
            },
            "cpu_stage0_debug_f32": {
                "manifest": manifest_for("python_reference_stage0_cpu_debug"),
                "device": CPU_DEVICE,
                "dtype": "float32",
                "case_ids": STAGE0_DEBUG_CASE_IDS,
                "mirrors": "gpu_stage0_debug_f32",
            },
            "cpu_dense_f32": {
                "manifest": manifest_for("python_reference_stage7_cpu_f32_dense"),
                "device": CPU_DEVICE,
                "dtype": "float32",
                "case_ids": DENSE_CASE_IDS,
                "mirrors": "gpu_dense_f32",
            },
        },
        "artifacts_root": str(v2_root / "artifacts"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze full CPU v2 baselines mirroring the GPU case set."
    )
    parser.add_argument("--v2-root", default=str(DEFAULT_V2_ROOT))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--ref-text-file", default=str(DEFAULT_REF_TEXT))
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="skip the large stage7 dense baseline (not recommended for full parity packs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v2_root = Path(args.v2_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    ref_audio = Path(args.ref_audio).resolve()
    ref_text_file = Path(args.ref_text_file).resolve()
    artifacts_root = v2_root / "artifacts"

    for path, label in (
        (model_dir, "model directory"),
        (ref_audio, "reference audio"),
        (ref_text_file, "reference text"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    # 1) product (same case IDs as GPU product)
    run_export(
        model_dir=model_dir,
        ref_audio=ref_audio,
        ref_text_file=ref_text_file,
        out_dir=artifacts_root / "python_reference_cpu_strict",
        case_ids=PRODUCT_CASE_IDS,
        debug_case_ids=PRODUCT_DEBUG_CASE_IDS,
        capture_steps=DEBUG_CAPTURE_STEPS,
        capture_layers=DEBUG_CAPTURE_LAYERS,
    )
    # 2) stage0 deterministic (same case IDs as GPU stage0)
    run_export(
        model_dir=model_dir,
        ref_audio=ref_audio,
        ref_text_file=ref_text_file,
        out_dir=artifacts_root / "python_reference_stage0_deterministic_cpu_strict",
        case_ids=STAGE0_CASE_IDS,
    )
    # 3) stage0 debug (mirror of python_reference_stage0_cuda_debug)
    run_export(
        model_dir=model_dir,
        ref_audio=ref_audio,
        ref_text_file=ref_text_file,
        out_dir=artifacts_root / "python_reference_stage0_cpu_debug",
        case_ids=STAGE0_DEBUG_CASE_IDS,
        debug_case_ids=STAGE0_DEBUG_CASE_IDS,
        capture_steps=DEBUG_CAPTURE_STEPS,
        capture_layers=DEBUG_CAPTURE_LAYERS,
    )
    # 4) dense stage0 (mirror of stage7 cuda dense)
    if not args.skip_dense:
        run_export(
            model_dir=model_dir,
            ref_audio=ref_audio,
            ref_text_file=ref_text_file,
            out_dir=artifacts_root / "python_reference_stage7_cpu_f32_dense",
            case_ids=DENSE_CASE_IDS,
            debug_case_ids=DENSE_CASE_IDS,
            capture_steps=CAPTURE_STEPS,
            capture_layers=DENSE_CAPTURE_LAYERS,
            capture_stage1_debug=True,
        )

    index_path = artifacts_root / "python_reference_v2_cpu_index.json"
    index = build_cpu_index(v2_root, model_dir, ref_audio, ref_text_file)
    if args.skip_dense:
        index["baselines"].pop("cpu_dense_f32", None)
    write_json(index_path, index)
    subprocess.run(
        [sys.executable, str(VALIDATE_REFERENCE), "--index", str(index_path)],
        check=True,
    )
    print(f"froze full CPU v2 baselines into {index_path}")


if __name__ == "__main__":
    main()
