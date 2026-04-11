# Phase 10 Stage0 Contract

## Input Contract
- Stage0 still consumes packed cond/uncond batches built from `PreparedPrompt`.
- Phase 10 keeps exact-token CUDA main-case parity as diagnostic-only when divergence is attributable to Candle numeric drift.

## Output Contract
- Primary acceptance for Stage0 is dense parity, token-domain validity, and successful downstream audio generation on GPU.
- Phase 7 parity suites are part of default GPU coverage in Phase 10; they are no longer opt-in via a dedicated test feature.
- Exact-token parity may remain ignored/diagnostic when proven to be Candle-specific numeric drift.

## Shape Contract
- No public tensor-shape changes from Phase 9.
- Generated tokens remain `(C, T)` or chunked lists of `(C, T)`.

## DType Contract
- `auto`/GPU execution prioritizes CUDA or Metal and uses runtime low precision where supported.
- Debug captures may materialize tensors on CPU for metric calculation only.

## Special Cases
- CPU Stage0 execution is not Phase 10 acceptance; it may remain as oracle/debug coverage only.
- No Candle patching is allowed. If a mismatch is isolated to Candle, only diagnostic thresholds or test classification may change, with evidence.
