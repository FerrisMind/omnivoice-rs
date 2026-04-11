# Phase 9 Frontend Contract

## Input Contract
- `RuntimeOptions::Auto` is GPU-first by contract: `cuda:0 -> metal -> cpu`.
- Frontend prompt assembly remains device-agnostic; primary acceptance executes it through CUDA/Metal pipelines.
- CPU exists only for explicit fallback, debug oracle, and host-side preprocessing/offload.

## Output Contract
- Frontend produces the same `PreparedPrompt` / chunking structure as Phase 8.
- Phase 9 acceptance requires `auto`, `design`, `clone`, `clone via ASR`, `long-form`, and `batch ordering` to succeed on GPU.

## Shape Contract
- No public tensor-shape changes from Phase 8.
- Chunked no-reference generation must keep chunk-0 text/tokens as the fixed reference for chunk 1+.

## DType Contract
- `auto` dtype resolves to `f16` on CUDA/Metal and `f32` only on CPU fallback.
- Frontend host-side strings, ids, and masks remain ordinary Rust-owned data until handed to runtime tensors.

## Special Cases
- CPU contract tests and CPU reference bundles remain diagnostic-only.
- Metal acceptance must mirror CUDA scenarios, but local Windows closeout is not blocked by lack of Metal hardware.
