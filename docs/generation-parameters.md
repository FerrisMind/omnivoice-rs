# Generation Parameters

This page documents the generation controls implemented by the Rust port.

These parameters are exposed through:

- `omnivoice-cli infer`
- `omnivoice-cli infer-batch`
- `GenerationConfig` in `crates/omnivoice-infer`
- the OpenAI-compatible server request extensions

## Defaults

Current defaults in the Rust port:

| Parameter | Default |
|---|---:|
| `num_step` | `32` |
| `guidance_scale` | `2.0` |
| `t_shift` | `0.1` |
| `layer_penalty_factor` | `5.0` |
| `position_temperature` | `5.0` |
| `class_temperature` | `0.0` |
| `denoise` | `true` |
| `preprocess_prompt` | `true` |
| `postprocess_output` | `true` |
| `audio_chunk_duration` | `15.0` |
| `audio_chunk_threshold` | `30.0` |

## Decoding

| Parameter | Type | Description |
|---|---|---|
| `num_step` | `usize` | Number of iterative unmasking steps. Higher values improve quality but cost latency. |
| `guidance_scale` | `f32` | Classifier-free guidance strength. |
| `t_shift` | `f32` | Time-step shift used in the schedule. |
| `layer_penalty_factor` | `f32` | Penalizes deeper codebook layers during unmask scheduling. |
| `position_temperature` | `f32` | Controls randomness of mask-position selection. |
| `class_temperature` | `f32` | Controls token sampling randomness. `0.0` is greedy. |

Practical presets:

- fastest deterministic-ish path:
  `num_step=16`, `class_temperature=0.0`
- default high-quality path:
  `num_step=32`, `guidance_scale=2.0`

## Prompt and Output Processing

| Parameter | Type | Description |
|---|---|---|
| `denoise` | `bool` | Adds the denoise signal for clone-oriented prompt handling. |
| `preprocess_prompt` | `bool` | Cleans prompt audio and normalizes reference text when preparing clone prompts. |
| `postprocess_output` | `bool` | Runs output post-processing after stage1 decode. |

Notes:

- `preprocess_prompt` matters most for clone mode
- `postprocess_output` affects final waveform cleanup
- `denoise=false` is exposed and tested by server-side request mapping

## Duration and Speed

| Parameter | Type | Description |
|---|---|---|
| `duration` | `Option<f32>` | Target duration in seconds. |
| `speed` | `Option<f32>` | Relative speed factor. Values above `1.0` shorten output. |

Priority:

- `duration` overrides `speed`
- if neither is set, the frontend estimates target length from prompt/reference heuristics

## Long-Form Chunking

| Parameter | Type | Description |
|---|---|---|
| `audio_chunk_duration` | `f32` | Target chunk duration when long-form chunking activates. |
| `audio_chunk_threshold` | `f32` | Estimated duration threshold above which chunking is enabled. |

Chunking is part of the implemented inference path and is used to keep VRAM/RAM usage more stable for long text.

## CLI Example

```powershell
cargo run -p omnivoice-cli -- infer `
  --model model `
  --text "Long-form synthesis example." `
  --language en `
  --output out\sample.wav `
  --device cpu `
  --dtype f32 `
  --num-step 16 `
  --guidance-scale 2.0 `
  --t-shift 0.1 `
  --audio-chunk-duration 15 `
  --audio-chunk-threshold 30
```

## Server Mapping

The OpenAI-compatible server accepts these as JSON request extensions under `/v1/audio/speech`.

Example:

```json
{
  "model": "default",
  "input": "hello",
  "voice": "alloy",
  "response_format": "wav",
  "num_step": 16,
  "guidance_scale": 2.0,
  "t_shift": 0.1,
  "denoise": false
}
```

## Current Caveats

- `dtype=auto` currently resolves to `f32`
- CPU inference is verified, but `--seed` on CPU is currently not dependable
- lower-precision GPU paths exist, but some parts of the pipeline intentionally stay on `f32` for stability
