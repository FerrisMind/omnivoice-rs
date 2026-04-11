# Phase 8 Frontend Contract

## Input Contract
- Frontend consumes `GenerationRequest` and normalizes all list-shaped fields to batch size.
- `voice_clone_prompts` override raw `ref_audios/ref_texts` for prompt construction.
- Language and instruct normalization are derived from the official OmniVoice upstream implementation only.

## Output Contract
- Frontend produces `GenerationTask` and per-item `PreparedPrompt`.
- Prompt text follows `_combine_text` semantics before tokenization.
- Long-form chunking is punctuation-aware and uses estimated audio-token density.

## Shape Contract
- Prepared prompt `input_ids` are `(1, C, S)`.
- Prompt `audio_mask` is `(1, S)`.
- Target audio span starts at `style_tokens + text_tokens + optional_reference_audio_tokens`.

## DType Contract
- Text/audio token IDs remain integer tensors.
- Prompt masks are boolean tensors before runtime conversion.

## Special Cases
- `_combine_text` replaces newlines with `.`, removes whitespace around CJK, removes whitespace before emotion tags, and preserves `[laughter]`.
- `add_punctuation` appends `.` or `。` only when the string lacks terminal punctuation.
- `resolve_instruct` cannot mix Chinese dialects with English accents and keeps mutually exclusive style categories disjoint.
- Auto mode falls back to `ref_text="Nice to meet you."` and `25` reference audio tokens for target-length estimation.
- Long-form chunking uses `audio_chunk_duration` and `audio_chunk_threshold` from generation config.
