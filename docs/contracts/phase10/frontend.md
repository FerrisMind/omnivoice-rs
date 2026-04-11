# Phase 10 Frontend Contract

## Input Contract
- `RuntimeOptions::Auto` remains GPU-first by contract: `cuda:0 -> metal -> cpu`.
- Frontend prompt assembly stays device-agnostic; primary acceptance executes it through CUDA/Metal pipelines.
- CPU exists only for explicit fallback, debug oracle, and host-side preprocessing/offload.

## Output Contract
- Frontend still produces the same `PreparedPrompt` / chunking structure as Phase 9.
- Phase 10 acceptance requires `auto`, `design`, `clone`, `clone via ASR`, `long-form`, `batch ordering`, and explicit `--device auto --dtype auto` inference to succeed on GPU-first runtime paths.
- `duration` overrides must match the official OmniVoice preprocessing contract:
  - estimate with `speed=1.0`
  - set `target_lens` from exact duration frame count
  - derive effective per-item speed as `estimated_tokens / target_lens`

## Shape Contract
- No public tensor-shape changes from Phase 9.
- Chunked no-reference generation must keep chunk-0 text/tokens as the fixed reference for chunk 1+.

## DType Contract
- `auto` dtype resolves to `f16` on CUDA/Metal and `f32` only on CPU fallback.
- Frontend host-side strings, ids, and masks remain ordinary Rust-owned data until handed to runtime tensors.

## Special Cases
- CPU contract tests and CPU reference bundles remain diagnostic-only.
- Metal acceptance must mirror CUDA scenarios, but local Windows closeout is not blocked by lack of Metal hardware.
