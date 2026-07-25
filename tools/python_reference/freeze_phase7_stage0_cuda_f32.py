#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPORT_REFERENCE = ROOT / "tools" / "python_reference" / "export_reference.py"
VALIDATE_REFERENCE = ROOT / "tools" / "python_reference" / "validate_reference.py"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "python_reference_stage7_cuda_f32_dense"
DEFAULT_INDEX_PATH = ROOT / "artifacts" / "python_reference_stage7_cuda_f32_dense_index.json"
DEFAULT_MODEL_DIR = ROOT / "model"
DEFAULT_REF_AUDIO = ROOT / "ref.wav"
DEFAULT_REF_TEXT = ROOT / "ref_text.txt"
CASE_IDS = [
    "det_auto_en_short",
    "det_design_en_british",
    "det_clone_user_ref",
    "det_auto_long_chunked",
]
CAPTURE_STEPS = ",".join(str(step) for step in range(32))
CAPTURE_LAYERS = ",".join([*(str(layer) for layer in range(28)), "final"])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_reference_index(out_dir: Path, seed: int) -> dict[str, Any]:
    artifacts_root = out_dir.parent
    return {
        "created_at": utc_now(),
        "root": str(ROOT),
        "seed": seed,
        "baselines": {
            "gpu_dense": {
                "manifest": str((out_dir / "manifest.json").relative_to(artifacts_root)).replace("\\", "/"),
                "device": "cuda:0",
                "dtype": "float32",
                "case_ids": CASE_IDS,
            }
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze dense Phase 7 OmniVoice stage0 GPU parity artifacts and validate them."
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--ref-text-file", default=str(DEFAULT_REF_TEXT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    ref_audio = Path(args.ref_audio).resolve()
    ref_text_file = Path(args.ref_text_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    index_path = Path(args.index_path).resolve()

    run_command(
        [
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
            "cuda:0",
            "--dtype",
            "float32",
            "--seed",
            str(args.seed),
            "--case-ids",
            ",".join(CASE_IDS),
            "--debug-case-ids",
            ",".join(CASE_IDS),
            "--debug-device",
            "cuda:0",
            "--debug-dtype",
            "float32",
            "--capture-steps",
            CAPTURE_STEPS,
            "--capture-layers",
            CAPTURE_LAYERS,
            "--capture-stage1-debug",
            "--gpu-only",
        ]
    )

    index = build_reference_index(out_dir, args.seed)
    write_json(index_path, index)

    run_command([sys.executable, str(VALIDATE_REFERENCE), "--index", str(index_path)])
    print(f"froze phase7 dense stage0 baseline into {index_path}")


if __name__ == "__main__":
    main()
