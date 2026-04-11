<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>OmniVoice Rust Port</b><br>
  GPU-first Rust workspace for OmniVoice inference, parity validation, and CLI execution with Candle.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-workspace-DEA584?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Candle-0.10.2-232323" alt="Candle">
  <img src="https://img.shields.io/badge/Inference-GPU--First-5B7CFA" alt="GPU First">
</p>

## 📚 Table of Contents

- [What is this?](#-what-is-this)
- [Demo](#-demo)
- [Key Features](#-key-features)
- [Installation & Setup](#️-installation--setup)
- [How to Start Using](#-how-to-start-using)
- [System Requirements](#️-system-requirements)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

## ✨ What is this?

This repository is a Rust workspace focused on porting OmniVoice inference to Candle with GPU-first execution.

It contains:

- `crates/omnivoice-infer` — the two-stage inference pipeline
- `crates/omnivoice-cli` — CLI for prompt prep, stage0/stage1 inspection, and full inference
- `crates/omnivoice-server` — separate OpenAI-compatible HTTP speech server binary
- `docs/contracts` — phase-by-phase behavior contracts
- `tools` — local reference and support scripts

The project treats CUDA and Metal as primary inference targets. CPU exists only for fallback, explicit offload, preprocessing, and debug materialization.

## 🎬 Demo

There is no public GUI demo in this workspace. The primary interface is the CLI.

```powershell
cargo run -p omnivoice-cli --features cuda -- infer `
  --text "Hello, this is a test of zero-shot text-to-speech." `
  --language en `
  --output H:\omnivoice\artifacts\demo.wav `
  --device cuda:0 `
  --dtype f32 `
  --seed 1234
```

If you want to force a local bundle or a specific Hugging Face repo, pass `--model`:

```powershell
cargo run -p omnivoice-cli --features cuda -- infer `
  --model H:\models\OmniVoice `
  --text "Hello from a local model bundle." `
  --output H:\omnivoice\artifacts\demo-local.wav
```

For an OpenAI-compatible speech endpoint, run the separate server binary:

```powershell
$env:OMNIVOICE_API_KEY="local-dev-token"
cargo run -p omnivoice-server --features cuda -- `
  --host 127.0.0.1 `
  --port 8000 `
  --device cuda:0 `
  --dtype f32
```

## 🚀 Key Features

- Two-stage OmniVoice inference pipeline in Rust
- GPU-first runtime selection: CUDA -> Metal -> CPU fallback
- Stage0 and Stage1 parity harnesses against official OmniVoice references
- Voice clone, voice design, auto voice, batch, and long-form chunked inference
- CLI workflows for `prepare-prompt`, `stage0-debug`, `stage1-decode`, and `infer`
- Separate `omnivoice-server` binary with `/v1/models` and `/v1/audio/speech`

### Hardware Acceleration

| Backend | Status | Notes |
|---------|:------:|-------|
| CPU | ✅ | Fallback, preprocessing, debug, offload |
| CUDA (NVIDIA) | ✅ | Primary local validation path |
| Metal (Apple) | ✅ | Implemented and mirrored in tests |

## 🛠️ Installation & Setup

### Prerequisites

- Rust toolchain
- For CUDA: NVIDIA GPU and compatible driver/toolkit
- For Metal: macOS with Metal support
- Either a local OmniVoice bundle or network access to download models into the Hugging Face cache
- Official upstream reference materials available locally

### Development

```powershell
cargo fmt
cargo test -p omnivoice-infer --features cuda --test phase_status -- --nocapture --test-threads=1
cargo test -p omnivoice-cli --features cuda --test phase10_cli_cuda -- --nocapture --test-threads=1
```

### Quality Checks

```powershell
cargo fmt
cargo clippy --workspace --all-targets
cargo test -p omnivoice-infer --features cuda --test phase10_cuda_acceptance -- --nocapture --test-threads=1
cargo test -p omnivoice-cli --features cuda --test phase10_cli_cuda -- --nocapture --test-threads=1
```

## 📖 How to Start Using

1. Either pass `--model <local-path>` or let the CLI/server auto-resolve `k2-fsa/OmniVoice` from Hugging Face.
2. If you omit `--asr-model`, Whisper defaults to `oxide-lab/whisper-large-v3-turbo-GGUF` and downloads the Candle-compatible `config.json`, `tokenizer.json`, and `whisper-large-v3-turbo-q4_0.gguf`.
3. Keep official upstream reference materials available locally for behavior verification.
4. Run sequential GPU tests instead of launching everything at once.
5. Use `omnivoice-cli infer` for end-to-end synthesis.

## 🖥️ System Requirements

- Windows, macOS, or Linux
- Enough RAM and VRAM for OmniVoice weights and runtime tensors
- For GPU acceleration:
  - NVIDIA GPU for CUDA
  - Apple Silicon / supported macOS GPU for Metal

## 🙏 Acknowledgments

This workspace builds on top of the official upstream projects:

- [OmniVoice](https://github.com/k2-fsa/OmniVoice)
- [candle](https://github.com/huggingface/candle)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)

Official upstream projects remain the source of truth for behavior and engineering reference, but they are not tracked as part of this repository.

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for focused upstream attribution relevant to this workspace.

## 📄 License

Apache-2.0 — see [LICENSE](LICENSE)

Copyright (c) 2026 FerrisMind
