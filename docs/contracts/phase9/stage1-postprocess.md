# Phase 9 Stage1 And Postprocess Contract

## Input Contract
- Stage1 decode remains driven by generated audio tokens plus optional `ref_rms`.
- GPU-first execution applies to full inference acceptance; host-side postprocess is allowed as explicit offload.

## Output Contract
- Stage1 raw and final decode outputs must still match reference sample rate and remain within established audio parity tolerances.
- Phase 9 acceptance requires successful GPU decode for `auto`, `design`, `clone`, `clone via ASR`, and `long-form`.

## Shape Contract
- Decoder input remains `(1, C, T)`.
- Final audio remains mono waveform output, with chunked paths merged into a single waveform.

## DType Contract
- Model compute must stay on the selected GPU backend.
- CPU transfers are allowed only for debug tensor materialization, host-owned return values, and explicit postprocess/offload helpers.

## Special Cases
- Silence removal, RMS restore, peak normalization, fade/pad, and chunk cross-fade remain correctness helpers, not alternative inference backends.
- Any remaining drift accepted here must be documented as diagnostic-only and justified with concrete CUDA/Metal evidence.
