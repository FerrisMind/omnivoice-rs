#!/usr/bin/env python3
"""Pack v2 fixture archives for a GitHub release.

Produces one or more repository-shaped zip assets (model weights excluded):

- ``ci-fixtures-v2-gpu.zip`` — GPU product / stage0 / dense baselines
- ``ci-fixtures-v2-cpu.zip`` — CPU-strict debug baselines
- optional combined ``ci-fixtures-v2.zip`` with both

Set ``OMNIVOICE_ROOT`` to the extracted root and place the model at ``./model``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_ROOT = ROOT / "artifacts" / "v2"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "dist"

GPU_BASELINES = (
    "python_reference",
    "python_reference_stage0_deterministic",
    "python_reference_stage0_cuda_debug",
    "python_reference_stage7_cuda_f32_dense",
)
CPU_BASELINES = (
    "python_reference_cpu_strict",
    "python_reference_stage0_deterministic_cpu_strict",
    "python_reference_stage0_cpu_debug",
    "python_reference_stage7_cpu_f32_dense",
)
EXTRA_ARTIFACT_DIRS = ("live_oracles",)
SKIP_DIR_NAMES = {"hook_test", "__pycache__"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


def build_portable_index(source_index: Path) -> dict[str, Any]:
    payload = json.loads(source_index.read_text(encoding="utf-8"))
    payload["created_at"] = utc_now()
    payload["root"] = "."
    payload["source_root"] = "omnivoice-rs"
    payload["model_dir"] = "model"
    payload["ref_audio"] = "ref.wav"
    payload["ref_text_file"] = "ref_text.txt"
    payload["artifacts_root"] = "artifacts"
    payload["packaged_at"] = utc_now()
    payload["includes_model"] = False
    return payload


def copy_baselines(artifacts_src: Path, artifacts_dst: Path, baselines: Sequence[str]) -> None:
    for baseline in baselines:
        src = artifacts_src / baseline
        if not src.is_dir():
            raise FileNotFoundError(f"missing baseline directory: {src}")
        shutil.copytree(
            src,
            artifacts_dst / baseline,
            ignore=shutil.ignore_patterns(*SKIP_DIR_NAMES, "*.pyc"),
        )


def copy_optional_dirs(artifacts_src: Path, artifacts_dst: Path, names: Sequence[str]) -> None:
    for name in names:
        src = artifacts_src / name
        if src.is_dir():
            shutil.copytree(
                src,
                artifacts_dst / name,
                ignore=shutil.ignore_patterns(*SKIP_DIR_NAMES, "*.pyc"),
            )


def write_readme(stage_root: Path, kind: str, baselines: Sequence[str], index_names: Sequence[str]) -> None:
    lines = [
        f"# OmniVoice CI fixtures v2 ({kind})",
        "",
        "Python golden references for `omnivoice-rs`.",
        "",
        "## Layout",
        "",
    ]
    for baseline in baselines:
        lines.append(f"- `artifacts/{baseline}`")
    for index_name in index_names:
        lines.append(f"- `artifacts/{index_name}`")
    lines.extend(
        [
            "- `ref.wav` / `ref_text.txt` — clone prompt assets",
            "- `artifacts/live_oracles/` — optional live clone oracle (when present)",
            "",
            "## Usage",
            "",
            "1. Extract this archive.",
            "2. Place or symlink the OmniVoice model at `./model`.",
            "3. Point tests at the extracted root:",
            "",
            "```powershell",
            "$env:OMNIVOICE_ROOT = (Resolve-Path .).Path",
            "cargo test -p omnivoice-infer --features 'cuda phase6-tests' -- --test-threads=1",
            "```",
            "",
            "Model weights are intentionally omitted from this package.",
            "",
        ]
    )
    (stage_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def stage_bundle(
    *,
    v2_root: Path,
    stage_root: Path,
    kind: str,
    baselines: Sequence[str],
    index_names: Sequence[str],
) -> Path:
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    for name in ("ref.wav", "ref_text.txt"):
        src = v2_root / name
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}")
        shutil.copy2(src, stage_root / name)

    artifacts_src = v2_root / "artifacts"
    artifacts_dst = stage_root / "artifacts"
    artifacts_dst.mkdir(parents=True)

    copy_baselines(artifacts_src, artifacts_dst, baselines)
    copy_optional_dirs(artifacts_src, artifacts_dst, EXTRA_ARTIFACT_DIRS)

    for index_name in index_names:
        index_src = artifacts_src / index_name
        if not index_src.is_file():
            raise FileNotFoundError(f"missing index: {index_src}")
        portable = build_portable_index(index_src)
        (artifacts_dst / index_name).write_text(
            json.dumps(portable, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    write_readme(stage_root, kind, baselines, index_names)
    return stage_root


def write_zip(stage_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in iter_files(stage_root):
            archive.write(path, arcname=str(path.relative_to(stage_root)).replace("\\", "/"))


def write_checksums(zip_path: Path) -> Path:
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    digest = sha256_file(zip_path)
    checksum_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return checksum_path


def pack_one(
    *,
    v2_root: Path,
    out_dir: Path,
    name: str,
    kind: str,
    baselines: Sequence[str],
    index_names: Sequence[str],
    keep_stage: bool,
) -> tuple[Path, Path]:
    stage_root = out_dir / f"{name}-stage"
    zip_path = out_dir / f"{name}.zip"
    stage_bundle(
        v2_root=v2_root,
        stage_root=stage_root,
        kind=kind,
        baselines=baselines,
        index_names=index_names,
    )
    write_zip(stage_root, zip_path)
    checksum_path = write_checksums(zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"packed {zip_path} ({size_mb:.1f} MiB)")
    print(f"sha256 {checksum_path}")
    if not keep_stage:
        shutil.rmtree(stage_root)
    return zip_path, checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack v2 fixtures for GitHub release.")
    parser.add_argument("--v2-root", default=str(DEFAULT_V2_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--set",
        choices=("gpu", "cpu", "both", "combined"),
        default="both",
        help="which archive set to build (default: separate gpu+cpu)",
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="keep staged directories next to the zip files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v2_root = Path(args.v2_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.set in ("gpu", "both"):
        pack_one(
            v2_root=v2_root,
            out_dir=out_dir,
            name="ci-fixtures-v2-gpu",
            kind="gpu",
            baselines=GPU_BASELINES,
            index_names=("python_reference_v2_index.json",),
            keep_stage=args.keep_stage,
        )
    if args.set in ("cpu", "both"):
        pack_one(
            v2_root=v2_root,
            out_dir=out_dir,
            name="ci-fixtures-v2-cpu",
            kind="cpu",
            baselines=CPU_BASELINES,
            index_names=("python_reference_v2_cpu_index.json",),
            keep_stage=args.keep_stage,
        )
    if args.set == "combined":
        pack_one(
            v2_root=v2_root,
            out_dir=out_dir,
            name="ci-fixtures-v2",
            kind="gpu+cpu",
            baselines=GPU_BASELINES + CPU_BASELINES,
            index_names=(
                "python_reference_v2_index.json",
                "python_reference_v2_cpu_index.json",
            ),
            keep_stage=args.keep_stage,
        )


if __name__ == "__main__":
    main()
