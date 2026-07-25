# Python Golden Reference

## GPU-only v2 bundle

The current fixture set is generated from the checked-out upstream Python
implementation and the local model with CUDA inference only. The v2 root is
repository-shaped so the Rust tests can use it directly:

```powershell
uv run --project 'C:/Users/PC/Documents/Default Project/omnivoice_refs/OmniVoice' `
  python tools/python_reference/freeze_v2_gpu.py
$env:OMNIVOICE_ROOT = (Resolve-Path artifacts/v2).Path
```

The command writes and validates:

- `artifacts/v2/artifacts/python_reference`
- `artifacts/v2/artifacts/python_reference_stage0_deterministic`
- `artifacts/v2/artifacts/python_reference_stage0_cuda_debug`
- `artifacts/v2/artifacts/python_reference_stage7_cuda_f32_dense`
- `artifacts/v2/artifacts/python_reference_v2_index.json`

`--gpu-only` rejects CPU case devices and disables the exporter's CPU
determinism fallback. Tensor validation may deserialize files to host memory,
but no model inference is performed on CPU.

### Freeze full CPU baselines (same case set as GPU)

```powershell
uv run --project 'C:/Users/PC/Documents/Default Project/omnivoice_refs/OmniVoice' `
  python tools/python_reference/freeze_v2_cpu.py
```

Writes (mirrors GPU baselines, CPU float32):

| CPU directory | Mirrors GPU |
|---|---|
| `python_reference_cpu_strict` | `python_reference` |
| `python_reference_stage0_deterministic_cpu_strict` | `python_reference_stage0_deterministic` |
| `python_reference_stage0_cpu_debug` | `python_reference_stage0_cuda_debug` |
| `python_reference_stage7_cpu_f32_dense` | `python_reference_stage7_cuda_f32_dense` |
| `python_reference_v2_cpu_index.json` | `python_reference_v2_index.json` |

Dense export on CPU is slow and large; omit only with `--skip-dense` if needed.

### Pack release assets (no model weights)

```powershell
python tools/python_reference/pack_v2_release.py --set both
# -> artifacts/dist/ci-fixtures-v2-gpu.zip (+ .sha256)
# -> artifacts/dist/ci-fixtures-v2-cpu.zip (+ .sha256)

# optional single combined archive:
python tools/python_reference/pack_v2_release.py --set combined
```

Extract a zip, place or symlink the OmniVoice model at `<root>/model`, then set
`OMNIVOICE_ROOT` to the extracted root.

This directory contains the Phase 0 harness that freezes the canonical Python reference for the future OmniVoice Rust port.

- source of truth: `h:/omnivoice/refs/OmniVoice`
- checkpoint: `h:/omnivoice/model`
- clone prompt assets: `h:/omnivoice/ref.wav`, `h:/omnivoice/ref_text.txt`

`candle` and `mistral.rs` are not behavioral sources of truth for these artifacts.

## Canonical run

From `h:/omnivoice`:

```powershell
uv run --project refs/OmniVoice python tools/python_reference/freeze_phase0.py
```

This command generates both baselines, writes the top-level index, and validates the result:

- `h:/omnivoice/artifacts/python_reference`
- `h:/omnivoice/artifacts/python_reference_cpu_strict`
- `h:/omnivoice/artifacts/python_reference_index.json`

## Baselines

### GPU product baseline

- output dir: `h:/omnivoice/artifacts/python_reference`
- device: `cuda:0`
- dtype: `float16`
- default case set: 6 cases

### CPU strict baseline

- output dir: `h:/omnivoice/artifacts/python_reference_cpu_strict`
- device: `cpu`
- dtype: `float32`
- case set: `debug_auto_en_short`, `debug_clone_user_ref`

## Exporter usage

Default GPU export:

```powershell
uv run --project refs/OmniVoice python tools/python_reference/export_reference.py `
  --model-dir h:/omnivoice/model `
  --ref-audio h:/omnivoice/ref.wav `
  --ref-text-file h:/omnivoice/ref_text.txt `
  --out-dir h:/omnivoice/artifacts/python_reference `
  --device cuda:0 `
  --dtype float16 `
  --seed 1234
```

Selective CPU strict export:

```powershell
uv run --project refs/OmniVoice python tools/python_reference/export_reference.py `
  --model-dir h:/omnivoice/model `
  --ref-audio h:/omnivoice/ref.wav `
  --ref-text-file h:/omnivoice/ref_text.txt `
  --out-dir h:/omnivoice/artifacts/python_reference_cpu_strict `
  --device cpu `
  --dtype float32 `
  --seed 1234 `
  --case-ids debug_auto_en_short,debug_clone_user_ref
```

## Artifact contract

Every export writes:

- `runtime.json`
- `manifest.json`
- `<case-id>/case.json`
- `<case-id>/prepared.json`
- `<case-id>/inputs.safetensors`
- `<case-id>/final_tokens.safetensors`
- `<case-id>/decoded_raw.wav`
- `<case-id>/final.wav`

Debug cases additionally write:

- `<case-id>/debug.json`
- `<case-id>/forward_step_00.safetensors`
- `<case-id>/steps/step_00.safetensors`
- `<case-id>/steps/step_15.safetensors`
- `<case-id>/steps/step_31.safetensors`

`prepared.json` now includes prompt-builder contract data needed for Rust parity:

- `style_text`
- `full_text`
- `style_token_ids`
- `text_token_ids`
- segment lengths and `target_start_idx`
- `prompt_contracts`
- chunk metadata for chunked runs

Debug `inputs.safetensors` additionally include:

- `batch_input_ids_before_step_00`
- `batch_audio_mask`
- `batch_attention_mask`
- `tokens_init`

## Phase 7 dense GPU baseline

Phase 7 uses a dedicated dense CUDA oracle baseline for stage0 parity:

- output dir: `h:/omnivoice/artifacts/python_reference_stage7_cuda_f32_dense`
- index: `h:/omnivoice/artifacts/python_reference_stage7_cuda_f32_dense_index.json`
- device: `cuda:0`
- dtype: `float32`
- cases:
  - `det_auto_en_short`
  - `det_design_en_british`
  - `det_clone_user_ref`
  - `det_auto_long_chunked`
- debug capture:
  - `capture_steps = 0..31`
  - `capture_layers = 0..27 + final`

Freeze it with:

```powershell
uv run --project refs/OmniVoice python tools/python_reference/freeze_phase7_stage0_cuda_f32.py
```

Validate it with:

```powershell
python tools/python_reference/validate_reference.py --index h:/omnivoice/artifacts/python_reference_stage7_cuda_f32_dense_index.json
```

## Validation

Validate any frozen dual-baseline index:

```powershell
python tools/python_reference/validate_reference.py --index h:/omnivoice/artifacts/python_reference_index.json
```

The validator checks:

- referenced files exist
- sha256 hashes match `manifest.json`
- final token tensors are `8 x T`
- `audio_mask_id=1024` does not remain in final tokens
- audio outputs are `24000 Hz`
- debug cases contain `debug.json`, `forward_step_00.safetensors`, and captured steps
- non-skipped determinism reports are marked `matched=true`
