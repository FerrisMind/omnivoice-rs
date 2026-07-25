# Listening demo — **Rust omnivoice-rs**

> Это **наш** проект (`omnivoice-cli`), **не** Python OmniVoice reference.
>
> Python-эталон, если нужен, лежит отдельно в `artifacts/listening_demo/`.

Сгенерировано: `tools/generate_listening_demo_rust.py`

## Как слушать

1. Открой `cuda/` (GPU), затем `cpu/`.
2. Одинаковые имена файлов = один сценарий на двух устройствах.
3. Рядом `.json` с командой CLI и статусом.
4. Сравнивай `cuda/NN_....wav` ↔ `cpu/NN_....wav`.

## Окружение

- CLI: `omnivoice-cli infer`
- model: local OmniVoice bundle (or `k2-fsa/OmniVoice`)
- ref audio: repo root `ref.wav` + `ref_text.txt`
- devices: `cuda_0`, `cpu`
- seed: `1234`
- created_at: `2026-07-25T11:34:39.992158Z`

Сравнение с Python-эталоном: [../LISTENING_COMPARE.md](../LISTENING_COMPARE.md).

> Папка `diag/` — локальная диагностика, в репозиторий не публикуется.

## Карта сценариев → файлы

| # | File | Mode | Сценарий | На что слушать |
|---|------|------|----------|----------------|
| 01 | `{device}/01_auto_en_short.wav` | `auto` | Auto voice — English short | Natural English auto voice from Rust. |
| 02 | `{device}/02_auto_zh_short.wav` | `auto` | Auto voice — Chinese short | Chinese auto voice. |
| 03 | `{device}/03_auto_lang_agnostic.wav` | `auto` | Auto voice — no language flag | Still intelligible without explicit language. |
| 04 | `{device}/04_design_en_british_female.wav` | `design` | Voice design — British female | Female British product-demo tone. |
| 05 | `{device}/05_design_en_male_american.wav` | `design` | Voice design — male American | Different from 04 — younger male American. |
| 06 | `{device}/06_design_zh_nonverbal.wav` | `design` | Voice design — Chinese + [laughter] | Laughter then Chinese line. |
| 07 | `{device}/07_clone_with_ref_text.wav` | `clone` | Voice clone — ref audio + ref text | Should match ref.wav speaker (ref text starts: “State-of-the-art text-to-speech model for 600+ languages, su…”). |
| 08 | `{device}/08_clone_asr_auto_transcript.wav` | `clone` | Voice clone — ref audio only (ASR) | Same speaker as 07 if ASR works in Rust CLI. |
| 09 | `{device}/09_long_chunked_en.wav` | `long` | Long-form chunked — English | Long continuous speech, smooth joins. |
| 10 | `{device}/10_control_speed_slow.wav` | `control` | Control — speed 0.85 | Slower than 01. |
| 11 | `{device}/11_control_speed_fast.wav` | `control` | Control — speed 1.2 | Faster than 01. |
| 12 | `{device}/12_control_fixed_duration.wav` | `control` | Control — fixed duration 4.5s | Duration near 4.5s in player. |
| 13 | `{device}/13_control_number_text.wav` | `control` | Control — numeric text | How Rust reads digits/dates. |
| 14 | `{device}/14_design_whisper.wav` | `design` | Voice design — whisper | Whispered / soft delivery. |
| 15 | `{device}/15_clone_same_as_07_seeded.wav` | `clone` | Voice clone — second seeded clone | Similar speaker to 07. |

`{device}` = `cuda` или `cpu`.

## Подробно

### `01_auto_en_short.wav` — Auto voice — English short

- **mode:** `auto`
- **description:** Rust infer: text + language only (no ref, no instruct).
- **language:** `English`
- **text:** OmniVoice creates clear speech from text with minimal setup.
- **listen for:** Natural English auto voice from Rust.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `02_auto_zh_short.wav` — Auto voice — Chinese short

- **mode:** `auto`
- **description:** Rust infer: Chinese language tag.
- **language:** `Chinese`
- **text:** 欢迎使用 OmniVoice。这是一段中文自动音色的试听样本。
- **listen for:** Chinese auto voice.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `03_auto_lang_agnostic.wav` — Auto voice — no language flag

- **mode:** `auto`
- **description:** Rust infer without --language.
- **language:** `None`
- **text:** This sample leaves language unspecified so the model chooses freely.
- **listen for:** Still intelligible without explicit language.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `04_design_en_british_female.wav` — Voice design — British female

- **mode:** `design`
- **description:** Rust infer with --instruct only.
- **language:** `English`
- **instruct:** `female, low pitch, british accent`
- **text:** Good afternoon. This reference should sound calm, precise, and suitable for a polished product demo.
- **listen for:** Female British product-demo tone.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `05_design_en_male_american.wav` — Voice design — male American

- **mode:** `design`
- **description:** Different supported instruct tags.
- **language:** `English`
- **instruct:** `male, high pitch, young adult, american accent`
- **text:** Hey there! Ready to ship another release? Let's make this one count.
- **listen for:** Different from 04 — younger male American.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `06_design_zh_nonverbal.wav` — Voice design — Chinese + [laughter]

- **mode:** `design`
- **description:** Chinese instruct (full-width commas) + nonverbal tag.
- **language:** `Chinese`
- **instruct:** `女，青年，中音调`
- **text:** [laughter]今天的发布会到此结束，感谢大家的聆听，祝你晚上愉快。
- **listen for:** Laughter then Chinese line.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `07_clone_with_ref_text.wav` — Voice clone — ref audio + ref text

- **mode:** `clone`
- **description:** Rust infer --ref-audio + --ref-text.
- **language:** `English`
- **clone:** ref_audio=yes, ref_text=yes
- **text:** This cloned sample should preserve the speaking style from the provided reference audio.
- **listen for:** Should match ref.wav speaker (ref text starts: “State-of-the-art text-to-speech model for 600+ languages, su…”).
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `08_clone_asr_auto_transcript.wav` — Voice clone — ref audio only (ASR)

- **mode:** `clone`
- **description:** Rust infer --ref-audio without --ref-text (ASR path if enabled).
- **language:** `English`
- **clone:** ref_audio=yes, ref_text=ASR/none
- **text:** Automatic transcription of the reference should still allow a solid clone.
- **listen for:** Same speaker as 07 if ASR works in Rust CLI.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `09_long_chunked_en.wav` — Long-form chunked — English

- **mode:** `long`
- **description:** Long text + --duration to force long/chunked path.
- **language:** `English`
- **duration:** 35.0
- **text:** OmniVoice can synthesize long-form speech while keeping memory usage stable. In this listening check, the text is intentionally longer than a typical demo sentence so the pipeline is forced through its chunking path. Each section should stay intelligible, maintain similar pacing, and join smoothly with the following section. We repeat the core idea so the model has enough text to split. OmniVoice can synthesize long-form speech while keeping memory usage stable.
- **listen for:** Long continuous speech, smooth joins.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `10_control_speed_slow.wav` — Control — speed 0.85

- **mode:** `control`
- **description:** --speed 0.85
- **language:** `English`
- **speed:** 0.85
- **text:** This sentence should be spoken more slowly than the default auto sample.
- **listen for:** Slower than 01.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `11_control_speed_fast.wav` — Control — speed 1.2

- **mode:** `control`
- **description:** --speed 1.2
- **language:** `English`
- **speed:** 1.2
- **text:** This sentence should be spoken faster than the default auto sample.
- **listen for:** Faster than 01.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `12_control_fixed_duration.wav` — Control — fixed duration 4.5s

- **mode:** `control`
- **description:** --duration 4.5
- **language:** `English`
- **duration:** 4.5
- **text:** Please stretch or compress this line to about four and a half seconds.
- **listen for:** Duration near 4.5s in player.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `13_control_number_text.wav` — Control — numeric text

- **mode:** `control`
- **description:** Digits/dates without optional TN package.
- **language:** `English`
- **text:** Please call me at 2345 on March 15, 2026 about invoice 99.
- **listen for:** How Rust reads digits/dates.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `14_design_whisper.wav` — Voice design — whisper

- **mode:** `design`
- **description:** Supported instruct: female, whisper, young adult.
- **language:** `English`
- **instruct:** `female, whisper, young adult`
- **text:** Keep this confidential. We will announce the launch next week.
- **listen for:** Whispered / soft delivery.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

### `15_clone_same_as_07_seeded.wav` — Voice clone — second seeded clone

- **mode:** `clone`
- **description:** Same clone inputs as 07 with same seed (Rust CLI has no separate create_voice_clone_prompt subcommand exposed the same way; this checks clone stability).
- **language:** `English`
- **clone:** ref_audio=yes, ref_text=yes
- **text:** Reusing clone settings should sound consistent with the first clone sample.
- **listen for:** Similar speaker to 07.
- **cuda_0:** ok, duration≈?
- **cpu:** ok, duration≈?

