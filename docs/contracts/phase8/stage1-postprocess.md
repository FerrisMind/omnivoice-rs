# Phase 8 Stage1 And Postprocess Contract

## Input Contract
- Stage1 accepts generated audio tokens as `(C, T)` or a chunked list of `(C, T)` tensors.
- Token IDs must remain inside `[token_id_min, token_id_max]` and use exactly `num_audio_codebooks` rows.

## Output Contract
- Stage1 raw decode returns mono `f32` waveform samples at the manifest sample rate.
- Final postprocess may remove long silences, restore clone RMS, peak-normalize auto voice, and apply fade/pad.

## Shape Contract
- Quantizer input is `(1, C, T)` after runtime preparation.
- Decoder output is a mono waveform; chunked outputs are merged into a single mono waveform.

## DType Contract
- Token tensors are integer IDs.
- Raw and final audio buffers are `f32`.
- PCM16-style silence detection and scaling are only an internal helper for parity with Python audio utilities.

## Special Cases
- Chunked decode uses cross-fade insertion compatible with the Python chunk merge policy.
- Clone output restores loudness only when `ref_rms < 0.1`.
- Auto-voice output peak-normalizes to `0.5` when no reference RMS is available.
- Fade-in/fade-out and silence padding always run after optional silence removal.
- Silence removal and PCM scaling are correctness helpers, not alternative inference paths.
