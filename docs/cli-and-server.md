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
  --output out\auto.wav
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
  --output out\clone.wav
```

### 3. Voice Design

No reference audio. The speaker is described with `--instruct`.

```powershell
cargo run -p omnivoice-cli -- infer `
  --model model `
  --text "This sample uses a designed voice." `
  --language en `
  --instruct "female, low pitch, british accent" `
  --output out\design.wav
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

- `auto` (default)
- `f32`
- `f16`
- `bf16`

`--dtype` is an optional debug override. You do not need it for normal inference. It does not pick which GGUF file to load and does not unpack quantized weights.

`auto` (default) chooses stage0 activations as:

- safetensors on CPU → `f32`
- GGUF on CPU → `f32` (Candle's quantized `f16` CPU kernel is scalar and much slower)
- safetensors on GPU → `bf16` if the weights are BF16, otherwise `f16`
- GGUF on GPU → `f16` (Candle `QMatMul` kernels accept `f32`/`f16` activations only, not `bf16`)

Weights stay in their file type. codec/stage1 stay `f32` until quantized conv is wired.

Current implementation notes:

- `auto` device prefers `CUDA -> Metal -> CPU`
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

Operational behavior:

- `GET /health` returns `200` only after the runtime is ready to serve synthesis requests
- during startup failures or model loading, `GET /health` returns `503`
- all routes can be mounted behind `--base-path`
- browser preflight requests are supported via CORS and `OPTIONS`

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
  --base-path /edge `
  --request-timeout-secs 300
```

## `POST /v1/audio/speech`

Base request fields:

- `model`
- `input`
- `voice`
- `response_format`

Transport:

- `application/json`
- `multipart/form-data`

For multipart requests, `ref_audio` can be uploaded as a real file instead of a base64 data URI.

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

Multipart notes:

- `ref_audio` is accepted as an uploaded audio file and decoded through the existing OmniVoice audio loader
- JSON requests remain backward-compatible and still accept `ref_audio` as a base64 data URI
- `voice` is accepted for OpenAI-shape compatibility but does not currently select a server-side voice catalog

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

Example multipart request:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer local-dev-token" \
  -F model=default \
  -F input="hello" \
  -F voice=alloy \
  -F language=en \
  -F ref_text="reference text" \
  -F ref_audio=@ref.wav \
  -F response_format=wav
```

Server behavior is tested against the OpenAI-compatible surface in [crates/omnivoice-server/tests/openai_server.rs](../crates/omnivoice-server/tests/openai_server.rs).
