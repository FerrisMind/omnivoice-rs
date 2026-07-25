# Listening comparison — Python reference vs Rust port

Side-by-side samples for the same prompts, seed (`1234`), and model weights.

| Set | Path | Generator |
|-----|------|-----------|
| **Python reference** (upstream OmniVoice) | [`listening_demo/`](listening_demo/) | `tools/python_reference/generate_listening_demo.py` |
| **Rust port** (`omnivoice-cli`) | [`listening_demo_rust/`](listening_demo_rust/) | `tools/generate_listening_demo_rust.py` |

## How to compare

1. Prefer **CUDA** samples first (`listening_demo/cuda/` vs `listening_demo_rust/cuda_0/`).
2. Match by scenario number `01`…`15` (same prompt / mode).
3. Then optionally compare CPU vs GPU within each implementation.
4. Per-file JSON next to each WAV records the exact call parameters.

Catalog details:

- [Python demo README](listening_demo/README.md)
- [Rust demo README](listening_demo_rust/README.md)

## CUDA A/B table

Click a link to open/download the WAV on GitHub.

| # | Mode | Scenario | Python (CUDA) | Rust (CUDA) |
|---|------|----------|---------------|-------------|
| 01 | auto | English short | [▶](listening_demo/cuda/01_auto_en_short.wav) | [▶](listening_demo_rust/cuda_0/01_auto_en_short.wav) |
| 02 | auto | Chinese short | [▶](listening_demo/cuda/02_auto_zh_short.wav) | [▶](listening_demo_rust/cuda_0/02_auto_zh_short.wav) |
| 03 | auto | Language-agnostic | [▶](listening_demo/cuda/03_auto_lang_agnostic.wav) | [▶](listening_demo_rust/cuda_0/03_auto_lang_agnostic.wav) |
| 04 | design | British female | [▶](listening_demo/cuda/04_design_en_british_female.wav) | [▶](listening_demo_rust/cuda_0/04_design_en_british_female.wav) |
| 05 | design | Male American | [▶](listening_demo/cuda/05_design_en_male_american.wav) | [▶](listening_demo_rust/cuda_0/05_design_en_male_american.wav) |
| 06 | design | Chinese + `[laughter]` | [▶](listening_demo/cuda/06_design_zh_nonverbal.wav) | [▶](listening_demo_rust/cuda_0/06_design_zh_nonverbal.wav) |
| 07 | clone | Ref audio + ref text | [▶](listening_demo/cuda/07_clone_with_ref_text.wav) | [▶](listening_demo_rust/cuda_0/07_clone_with_ref_text.wav) |
| 08 | clone | Ref audio only (ASR) | [▶](listening_demo/cuda/08_clone_asr_auto_transcript.wav) | [▶](listening_demo_rust/cuda_0/08_clone_asr_auto_transcript.wav) |
| 09 | long | Long-form chunked EN | [▶](listening_demo/cuda/09_long_chunked_en.wav) | [▶](listening_demo_rust/cuda_0/09_long_chunked_en.wav) |
| 10 | control | Speed 0.85× | [▶](listening_demo/cuda/10_control_speed_slow.wav) | [▶](listening_demo_rust/cuda_0/10_control_speed_slow.wav) |
| 11 | control | Speed 1.2× | [▶](listening_demo/cuda/11_control_speed_fast.wav) | [▶](listening_demo_rust/cuda_0/11_control_speed_fast.wav) |
| 12 | control | Fixed duration 4.5s | [▶](listening_demo/cuda/12_control_fixed_duration.wav) | [▶](listening_demo_rust/cuda_0/12_control_fixed_duration.wav) |
| 13 | control | Numeric text | [▶](listening_demo/cuda/13_control_number_text.wav) | [▶](listening_demo_rust/cuda_0/13_control_number_text.wav) |
| 14 | design | Whisper | [▶](listening_demo/cuda/14_design_whisper.wav) | [▶](listening_demo_rust/cuda_0/14_design_whisper.wav) |
| 15 | clone | Clone reuse / seeded repeat | [▶](listening_demo/cuda/15_clone_then_reuse_prompt.wav) | [▶](listening_demo_rust/cuda_0/15_clone_same_as_07_seeded.wav) |

Scenario **15** is intentionally not identical across stacks: Python exercises `create_voice_clone_prompt` reuse; Rust checks a second seeded clone with the same inputs as 07.

## CPU samples

Same filenames under:

- [`listening_demo/cpu/`](listening_demo/cpu/)
- [`listening_demo_rust/cpu/`](listening_demo_rust/cpu/)

## Notes

- Sample rate: 24 kHz PCM WAV.
- Clone scenarios use the repo root `ref.wav` / `ref_text.txt`.
- These clips are qualitative listening checks, not a numerical parity score.
- Other local `artifacts/` paths remain gitignored (parity dumps, scratch exports, etc.).
