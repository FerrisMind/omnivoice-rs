#!/usr/bin/env python3
"""Run GGUF/Rust bench scenarios one-by-one; stop on first failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bench_all_scenarios as b  # noqa: E402

OUT = ROOT / "artifacts" / "bench_scenarios_gguf_q4km_cpu"
KEY = "rust_cpu_auto"
REF_TEXT = "State-of-the-art text-to-speech model for 600+ languages, supporting"


def main() -> int:
    path = OUT / "results.json"
    if path.exists():
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        model = (ROOT / "artifacts" / "gguf-q4km").resolve()
        results = {
            "created_at": b.utc_now(),
            "seed": 1234,
            "repeats": 1,
            "model": str(model),
            "hardware": "CPU",
            "dtypes": ["auto"],
            "runs": {},
        }

    block = results["runs"].setdefault(KEY, {"rows": []})
    done = {r["scenario_id"] for r in block.get("rows", []) if r.get("ok")}
    pending = [s for s in b.build_scenarios() if s.id not in done]
    if not pending:
        print("All scenarios already completed.", flush=True)
        return 0

    sc = pending[0]
    print(f"Running {sc.id} ({sc.title})...", flush=True)
    data = b.run_rust(
        model_dir=Path(results["model"]),
        ref_audio=(ROOT / "ref.wav").resolve(),
        ref_text=REF_TEXT,
        device="cpu",
        seed=1234,
        scenarios=[sc],
        repeats=1,
        release=True,
        dtype="auto",
    )
    if not data.get("ok") or not data.get("rows"):
        print(f"FAILED setup: {data.get('error')}", flush=True)
        tail = data.get("stdout_tail") or ""
        if tail:
            print(tail[-2000:], flush=True)
        return 1

    row = data["rows"][0]
    if not (row.get("ok") and row.get("wall_s") and row.get("audio_s")):
        print(f"FAILED {sc.id}: {row.get('error')}", flush=True)
        return 1

    for k, v in data.items():
        if k != "rows":
            block[k] = v
    block["rows"] = [r for r in block.get("rows", []) if r.get("scenario_id") != sc.id]
    block["rows"].extend(data["rows"])
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"OK {sc.id}: wall_s={row['wall_s']:.1f}s audio_s={row['audio_s']} "
        f"RTF={row.get('rtf')} ({len(block['rows'])}/15 done)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
