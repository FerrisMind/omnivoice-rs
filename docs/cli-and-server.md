# CLI and Server

The OmniVoice Rust port ships two primary user-facing binaries:

- `omnivoice-cli`
- `omnivoice-server`

## CLI Overview

The CLI supports:

- full end-to-end inference
- prompt preparation and inspection
- stage0 token generation
- stage1 decoding
- batch inference

Main commands:

- `infer`
- `infer-batch`
- `prepare-prompt`
- `stage1-prepare`
- `stage1-decode`
- `stage0-generate`
- `stage0-debug`
- `artifacts validate`

## Common Inference Modes

### 1. Auto Voice

No reference audio and no voice-design instruction. The model chooses a voice automatically.

```powershell
cargo run -p omnivoice-cli -- infer `
  --model model `
  --text "Hello, this is an auto voice example." `
  --language en `
  --output out\auto.wav `
  --device cpu `
  --dtype f32
```

### 2. Voice Clone

Reference audio plus reference text, or reference audio with ASR fallback when `--ref-text` is omitted.

```powershell
cargo run -p omnivoice-cli -- infer `
  --model model `
  --text "This sample should preserve the reference speaking style." `
  --language en `
  --ref-audio ref.wav `
  --ref-text (Get-Content ref_text.txt -Raw) `
  --output out\clone.wav `
  --device cpu `
  --dtype f32
```

### 3. Voice Design

No reference audio. The speaker is described with `--instruct`.

```powershell
cargo run -p omnivoice-cli -- infer `
  --model model `
  --text "This sample uses a designed voice." `
  --language en `
  --instruct "female, low pitch, british accent" `
  --output out\design.wav `
  --device cpu `
  --dtype f32
```

## Device and DType Selection

Accepted device values:

- `auto`
- `cpu`
- `cuda`
- `cuda:N`
- `mps`
- `metal`

Accepted dtype values:

- `auto`
- `f32`
- `f16`
- `bf16`

Current implementation notes:

- `auto` device prefers `CUDA -> Metal -> CPU`
- `auto` dtype currently resolves to `f32`
- CPU-only inference is supported
- GPU acceleration is optional but preferred when available

## Important CPU Note About `--seed`

CPU inference works, but `--seed` is currently not a reliable CPU option in this workspace.

Observed local behavior:

- CPU inference without `--seed`: works
- CPU inference with `--seed`: can fail with `Candle(cannot seed the CPU rng with set_seed)`

Until that is fixed in the port or upstream backend behavior changes, omit `--seed` for CPU runs.

## Batch Inference

`infer-batch` reads a JSONL test list and writes generated WAV files into a result directory.

Useful controls:

- `--batch-size`
- `--batch-duration`
- `--nj-per-gpu`
- `--warmup`

Use this when you want to process multiple samples under one runtime configuration.

## OpenAI-Compatible Server

`omnivoice-server` exposes:

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`

Authentication:

- required via `Authorization: Bearer <token>`
- token source: `--api-key` or `OMNIVOICE_API_KEY`

Start the server:

```powershell
$env:OMNIVOICE_API_KEY = "local-dev-token"
cargo run -p omnivoice-server -- `
  --model model `
  --host 127.0.0.1 `
  --port 8000 `
  --device auto `
  --dtype auto
```

## `POST /v1/audio/speech`

Base request fields:

- `model`
- `input`
- `voice`
- `response_format`

Supported `response_format` values:

- `wav`
- `pcm`
- `mp3`

Optional request extensions accepted by this server:

- `language`
- `duration`
- `speed`
- `ref_text`
- `ref_audio` as a base64 data URI
- `instruct`
- `asr_model`
- `seed`
- `num_step`
- `guidance_scale`
- `t_shift`
- `layer_penalty_factor`
- `position_temperature`
- `class_temperature`
- `preprocess_prompt`
- `postprocess_output`
- `denoise`
- `audio_chunk_duration`
- `audio_chunk_threshold`
- `stream_format`

Example request:

```json
{
  "model": "default",
  "input": "hello",
  "voice": "alloy",
  "response_format": "wav",
  "language": "en",
  "speed": 1.0,
  "instruct": "female, low pitch, british accent",
  "num_step": 16
}
```

Server behavior is tested against the OpenAI-compatible surface in [crates/omnivoice-server/tests/openai_server.rs](../crates/omnivoice-server/tests/openai_server.rs).
