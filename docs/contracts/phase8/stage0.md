# Phase 8 Stage0 Contract

## Input Contract
- Stage0 consumes packed cond/uncond batches created from `PreparedPrompt`.
- Cond branch carries full prompt context; uncond branch carries only the target-audio tail.
- `audio_mask_id` is the only legal masked audio token and must never be reintroduced after selection.

## Output Contract
- Stage0 returns generated audio-token tensors `(C, T)` or chunked lists of them.
- Debug runs additionally expose forward-pass and per-step tensors for parity against Python captures.

## Shape Contract
- `batch_input_ids`: `(2B, C, max_seq_len)`
- `batch_audio_mask`: `(2B, max_seq_len)`
- `batch_attention_mask`: `(2B, 1, max_seq_len, max_seq_len)`
- Runtime logits: `(B, C, target_len, vocab)`

## DType Contract
- Token tensors remain integer IDs.
- Attention and audio masks are boolean before conversion to runtime tensors.
- CFG and sampling operate on `f32` logits even when runtime weights run on GPU with lower precision.

## Special Cases
- Audio embeddings are shifted by codebook offsets before selection.
- Unconditional padding keeps the padding diagonal visible in the attention mask.
- CFG follows `c + scale * (c - u)` before `log_softmax`.
- `audio_mask_id` is always forced to `-inf` before token selection.
- Long-form no-reference generation reuses chunk 0 tokens and chunk 0 text as the fixed reference for chunk 1+.
- Long-form clone generation preserves the original reference tokens/text for every chunk.
