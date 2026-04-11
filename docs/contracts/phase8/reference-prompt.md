# Phase 8 Reference Prompt Contract

## Input Contract
- Input is either a file path or an in-memory waveform.
- Waveform is downmixed to mono before any prompt-specific processing.
- Input is resampled to the model sample rate before RMS or tokenization.

## Output Contract
- Output is a reusable clone prompt payload: waveform samples, original `ref_rms`, and optional normalized `ref_text`.
- The returned waveform is always trimmed to a multiple of `hop_length`.

## Shape Contract
- Reference prompt waveform is a mono 1-D sample vector.
- Encoded reference prompt tokens must be `(C, T)` where `C == num_audio_codebooks`.

## DType Contract
- Runtime audio samples are `f32`.
- Audio tokens are integer IDs after tokenizer encode.

## Special Cases
- RMS boost applies only when `0 < ref_rms < 0.1`; stored `ref_rms` remains the original pre-boost RMS.
- With `preprocess_prompt=true` and missing `ref_text`, long prompt audio may be trimmed before silence removal.
- With `preprocess_prompt=true` and user-provided `ref_text`, long prompt trimming is skipped, but silence removal still runs.
- Punctuation is appended to `ref_text` only when `preprocess_prompt=true`.
- After preprocessing, prompt audio is truncated to a `hop_length` multiple before audio-tokenizer encode.
- If `ref_text` is absent after preprocessing, the ASR path owns transcript recovery.
