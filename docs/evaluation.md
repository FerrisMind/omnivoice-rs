# Evaluation and Verification

This repository is an inference port, so the evaluation story is centered on parity, reproducibility, and artifact-backed verification rather than training benchmarks.

## What Is Verified Here

This repository verifies:

- prompt construction and frontend behavior
- stage0 token generation behavior
- stage1 decoding and postprocess behavior
- CPU and GPU inference paths
- parity against upstream-generated reference artifacts

## Artifact Roots

Important artifact directories in this repository:

- `artifacts/python_reference/`
- `artifacts/python_reference_cpu_strict/`
- `artifacts/python_reference_stage0_deterministic/`
- `artifacts/python_reference_stage7_cuda_f32_dense/`
- `artifacts/live_oracles/`
- `artifacts/manual_cuda_modes/`

## Key Manifest Files

- [`artifacts/python_reference/manifest.json`](../artifacts/python_reference/manifest.json)
  Main upstream-style reference artifact set, including main CUDA cases.
- [`artifacts/python_reference_cpu_strict/manifest.json`](../artifacts/python_reference_cpu_strict/manifest.json)
  CPU strict/debug verification artifacts.
- [`artifacts/python_reference_stage7_cuda_f32_dense/manifest.json`](../artifacts/python_reference_stage7_cuda_f32_dense/manifest.json)
  Dense CUDA stage0 parity artifacts.

## Verified Modes

Verified by existing artifacts:

- GPU `auto`
- GPU `clone`
- GPU `design`
- GPU long-form chunked generation
- CPU debug/reference flows

Verified locally in this workspace session:

- CPU `auto`
- CPU `clone`
- CPU `design`

## Useful Commands

### Validate artifact bundle wiring

```powershell
cargo run -p omnivoice-cli -- artifacts validate --model model
```

### CPU smoke inference

```powershell
cargo run -p omnivoice-cli -- infer `
  --model model `
  --text "This is a CPU verification run." `
  --language en `
  --output out\cpu.wav `
  --device cpu `
  --dtype f32
```

### CUDA inference

```powershell
cargo run -p omnivoice-cli --features cuda -- infer `
  --model model `
  --text "This is a CUDA verification run." `
  --language en `
  --output out\cuda.wav `
  --device cuda:0 `
  --dtype f32
```

### Metal inference

```powershell
cargo run -p omnivoice-cli --features metal -- infer `
  --model model `
  --text "This is a Metal verification run." `
  --language en `
  --output out\metal.wav `
  --device metal `
  --dtype f32
```

## Test Suites

Representative tests live under:

- `crates/omnivoice-infer/tests/`
- `crates/omnivoice-cli/tests/`
- `crates/omnivoice-server/tests/`

Useful examples:

- phase acceptance tests for CUDA and Metal
- server compatibility tests for `/v1/models` and `/v1/audio/speech`
- ASR selection and remote/local model resolution tests

## Behavior Contracts

Advanced expected behavior is documented in:

- [contracts/README.md](./contracts/README.md)

Use the contracts docs when you need phase-by-phase assertions rather than user-facing usage guidance.

## Current Caveats

- CPU inference is verified, but `--seed` on CPU is currently not dependable
- GPU validation is stronger for CUDA than for Metal because more reference artifacts are checked in for CUDA
- some debug and parity artifacts are diagnostic-only and should not be mistaken for end-user benchmark suites
