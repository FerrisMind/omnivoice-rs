# Phase 9 Reference Prompt Contract

## Input Contract
- `ReferenceAudioInput::FilePath` accepts common audio files through crates.io decoding, not only WAV.
- `(waveform, sample_rate)` remains a valid direct input path and is unchanged.

## Output Contract
- Reference prompt creation must produce the same voice-clone prompt structure as the official OmniVoice path: normalized mono waveform, optional ASR text, audio tokens, and `ref_rms`.

## Shape Contract
- Output audio tokens remain `(C, T)` with `C == num_audio_codebooks`.
- Host-side waveform preprocessing must trim to a hop-length multiple before tokenizer encode.

## DType Contract
- File decode and preprocessing operate on host-side `f32` waveform buffers.
- Returning tokens to Rust host structs is allowed and is not treated as CPU inference fallback.

## Special Cases
- Preprocess order must match official behavior closely: mono mix, resample, quiet-reference RMS boost, long-audio trim by silence gap, silence removal, hop trim.
- If parity gaps remain here, they must be documented as input-path limitations, not hidden behind fake success paths.
