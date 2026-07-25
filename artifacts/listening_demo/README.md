# OmniVoice listening demo

Ручная прослушка эталона **Python OmniVoice** (upstream reference).

Сгенерировано скриптом `tools/python_reference/generate_listening_demo.py`.

## Как слушать

1. Сначала папка `cuda/` (если есть), потом `cpu/`.
2. Имена файлов = `NN_scenario_id.wav` — одинаковые на обоих устройствах.
3. Рядом с каждым WAV лежит `NN_scenario_id.json` с параметрами вызова.
4. Сравнивайте **один и тот же номер** между CUDA и CPU.

## Окружение

- model: local OmniVoice bundle (or `k2-fsa/OmniVoice`)
- ref audio (clone): repo root `ref.wav` + `ref_text.txt`
- devices run: `cuda`, `cpu`
- created_at: `2026-07-25T10:28:29.821173Z`
- seed: `1234`
- sampling rate: `24000 Hz`, PCM_16 WAV

Сравнение с Rust-портом: [../LISTENING_COMPARE.md](../LISTENING_COMPARE.md).

## Карта сценариев → файлы

| # | File | Mode | What it tests | What to listen for |
|---|------|------|---------------|--------------------|
| 01 | `{device}/01_auto_en_short.wav` | `auto` | Auto voice — English short | Natural English voice, clear phrasing, no clone of ref.wav. |
| 02 | `{device}/02_auto_zh_short.wav` | `auto` | Auto voice — Chinese short | Chinese auto voice, not English clone. |
| 03 | `{device}/03_auto_lang_agnostic.wav` | `auto` | Auto voice — language-agnostic | Should still be intelligible English-ish speech without explicit lang. |
| 04 | `{device}/04_design_en_british_female.wav` | `design` | Voice design — British female | Female British accent, calm product-demo tone. |
| 05 | `{device}/05_design_en_male_american.wav` | `design` | Voice design — male American high pitch | Clearly different from 04 — younger male American voice. |
| 06 | `{device}/06_design_zh_nonverbal.wav` | `design` | Voice design — Chinese + nonverbal tag | Brief laughter then Chinese closing line. |
| 07 | `{device}/07_clone_with_ref_text.wav` | `clone` | Voice clone — ref audio + ref text | Should match speaker/style of ref.wav (ref text: “State-of-the-art text-to-speech model for 600+ languages, supporting…”). |
| 08 | `{device}/08_clone_asr_auto_transcript.wav` | `clone` | Voice clone — ref audio only (ASR transcript) | Same speaker as 07 if ASR succeeds; may fail if ASR deps missing. |
| 09 | `{device}/09_long_chunked_en.wav` | `long` | Long-form chunked — English | Long continuous speech, smooth joins, no harsh cuts between chunks. |
| 10 | `{device}/10_control_speed_slow.wav` | `control` | Control — slow speed (0.85×) | Noticeably slower than 01_auto_en_short. |
| 11 | `{device}/11_control_speed_fast.wav` | `control` | Control — fast speed (1.2×) | Noticeably faster than 01_auto_en_short. |
| 12 | `{device}/12_control_fixed_duration.wav` | `control` | Control — fixed duration (4.5 s) | Clip length near 4.5s (check WAV duration in player). |
| 13 | `{device}/13_control_number_text.wav` | `control` | Control — numeric text (no external TN) | How digits/dates are spoken without optional text-normalization deps. |
| 14 | `{device}/14_design_whisper.wav` | `design` | Voice design — whisper | Quiet / whispered quality. |
| 15 | `{device}/15_clone_then_reuse_prompt.wav` | `clone` | Voice clone — create_voice_clone_prompt reuse | Similar speaker to 07_clone_with_ref_text. |

Подставьте `device` = `cuda` или `cpu`.

## Подробно по сценариям

### `01_auto_en_short.wav` — Auto voice — English short

- **mode:** `auto`
- **description:** Neither reference audio nor style instruct. Model picks a voice. Language is explicitly English.
- **language:** `English`
- **text:** OmniVoice creates clear speech from text with minimal setup.
- **listen for:** Natural English voice, clear phrasing, no clone of ref.wav.

- **cuda:** ok, duration≈3.61s, peak=0.500
- **cpu:** ok, duration≈3.20s, peak=0.500

### `02_auto_zh_short.wav` — Auto voice — Chinese short

- **mode:** `auto`
- **description:** Auto voice with Chinese language tag.
- **language:** `Chinese`
- **text:** 欢迎使用 OmniVoice。这是一段中文自动音色的试听样本。
- **listen for:** Chinese auto voice, not English clone.

- **cuda:** ok, duration≈4.96s, peak=0.480
- **cpu:** ok, duration≈4.96s, peak=0.500

### `03_auto_lang_agnostic.wav` — Auto voice — language-agnostic

- **mode:** `auto`
- **description:** language=None (language-agnostic mode).
- **language:** `None`
- **text:** This sample leaves language unspecified so the model chooses freely.
- **listen for:** Should still be intelligible English-ish speech without explicit lang.

- **cuda:** ok, duration≈4.02s, peak=0.500
- **cpu:** ok, duration≈3.71s, peak=0.500

### `04_design_en_british_female.wav` — Voice design — British female

- **mode:** `design`
- **description:** Style control via instruct text only (no reference audio).
- **language:** `English`
- **instruct:** `female, low pitch, british accent`
- **text:** Good afternoon. This reference should sound calm, precise, and suitable for a polished product demo.
- **listen for:** Female British accent, calm product-demo tone.

- **cuda:** ok, duration≈6.12s, peak=0.500
- **cpu:** ok, duration≈5.49s, peak=0.500

### `05_design_en_male_american.wav` — Voice design — male American high pitch

- **mode:** `design`
- **description:** Different instruct attributes using only OmniVoice-supported English tags (see voice-design docs).
- **language:** `English`
- **instruct:** `male, high pitch, young adult, american accent`
- **text:** Hey there! Ready to ship another release? Let's make this one count.
- **listen for:** Clearly different from 04 — younger male American voice.

- **cuda:** ok, duration≈4.24s, peak=0.500
- **cpu:** ok, duration≈4.24s, peak=0.500

### `06_design_zh_nonverbal.wav` — Voice design — Chinese + nonverbal tag

- **mode:** `design`
- **description:** Chinese text with [laughter] control tag. Instruct uses full-width Chinese attribute list.
- **language:** `Chinese`
- **instruct:** `女，青年，中音调`
- **text:** [laughter]今天的发布会到此结束，感谢大家的聆听，祝你晚上愉快。
- **listen for:** Brief laughter then Chinese closing line.

- **cuda:** ok, duration≈5.72s, peak=0.500
- **cpu:** ok, duration≈5.80s, peak=0.500

### `07_clone_with_ref_text.wav` — Voice clone — ref audio + ref text

- **mode:** `clone`
- **description:** Classic zero-shot clone: provide ref.wav and its transcript. Target text is different from the reference transcript.
- **language:** `English`
- **clone:** ref_audio=yes, ref_text=yes
- **text:** This cloned sample should preserve the speaking style from the provided reference audio.
- **listen for:** Should match speaker/style of ref.wav (ref text: “State-of-the-art text-to-speech model for 600+ languages, supporting…”).

- **cuda:** ok, duration≈4.34s, peak=0.467
- **cpu:** ok, duration≈4.45s, peak=0.417

### `08_clone_asr_auto_transcript.wav` — Voice clone — ref audio only (ASR transcript)

- **mode:** `clone`
- **description:** Clone with ref_audio but without ref_text. OmniVoice ASR auto-transcribes the reference clip.
- **language:** `English`
- **clone:** ref_audio=yes, ref_text=ASR/auto
- **text:** Automatic transcription of the reference should still allow a solid clone.
- **listen for:** Same speaker as 07 if ASR succeeds; may fail if ASR deps missing.

- **cuda:** ok, duration≈4.01s, peak=0.420
- **cpu:** ok, duration≈4.01s, peak=0.469

### `09_long_chunked_en.wav` — Long-form chunked — English

- **mode:** `long`
- **description:** Long text forces chunked generation (audio_chunk_threshold / duration). Checks stitching and pacing across chunks.
- **language:** `English`
- **duration:** `35.0` s
- **text:** OmniVoice can synthesize long-form speech while keeping memory usage stable. In this listening check, the text is intentionally longer than a typical demo sentence so the pipeline is forced through its chunking path. Each section should stay intelligible, maintain similar pacing, and join smoothly with the following section. We repeat the core idea so the model has enough text to split. OmniVoice can synthesize long-form speech while keeping memory usage stable.
- **listen for:** Long continuous speech, smooth joins, no harsh cuts between chunks.

- **cuda:** ok, duration≈38.86s, peak=0.500
- **cpu:** ok, duration≈37.14s, peak=0.500

### `10_control_speed_slow.wav` — Control — slow speed (0.85×)

- **mode:** `control`
- **description:** Auto English with speed=0.85 (slower speech).
- **language:** `English`
- **speed:** `0.85`
- **text:** This sentence should be spoken more slowly than the default auto sample.
- **listen for:** Noticeably slower than 01_auto_en_short.

- **cuda:** ok, duration≈4.00s, peak=0.500
- **cpu:** ok, duration≈3.85s, peak=0.500

### `11_control_speed_fast.wav` — Control — fast speed (1.2×)

- **mode:** `control`
- **description:** Auto English with speed=1.2 (faster speech).
- **language:** `English`
- **speed:** `1.2`
- **text:** This sentence should be spoken faster than the default auto sample.
- **listen for:** Noticeably faster than 01_auto_en_short.

- **cuda:** ok, duration≈3.62s, peak=0.500
- **cpu:** ok, duration≈3.64s, peak=0.500

### `12_control_fixed_duration.wav` — Control — fixed duration (4.5 s)

- **mode:** `control`
- **description:** duration=4.5 forces approximate output length (overrides speed).
- **language:** `English`
- **duration:** `4.5` s
- **text:** Please stretch or compress this line to about four and a half seconds.
- **listen for:** Clip length near 4.5s (check WAV duration in player).

- **cuda:** ok, duration≈3.72s, peak=0.500
- **cpu:** ok, duration≈4.08s, peak=0.500

### `13_control_number_text.wav` — Control — numeric text (no external TN)

- **mode:** `control`
- **description:** Speaks a line with digits without WeTextProcessing (normalize_text=False). Shows default number reading.
- **language:** `English`
- **text:** Please call me at 2345 on March 15, 2026 about invoice 99.
- **listen for:** How digits/dates are spoken without optional text-normalization deps.

- **cuda:** ok, duration≈5.72s, peak=0.500
- **cpu:** ok, duration≈5.72s, peak=0.500

### `14_design_whisper.wav` — Voice design — whisper

- **mode:** `design`
- **description:** Instruct uses only supported tags: female + whisper.
- **language:** `English`
- **instruct:** `female, whisper, young adult`
- **text:** Keep this confidential. We will announce the launch next week.
- **listen for:** Quiet / whispered quality.

- **cuda:** ok, duration≈4.00s, peak=0.500
- **cpu:** ok, duration≈3.95s, peak=0.500

### `15_clone_then_reuse_prompt.wav` — Voice clone — create_voice_clone_prompt reuse

- **mode:** `clone`
- **description:** Builds VoiceClonePrompt once via create_voice_clone_prompt, then generates with voice_clone_prompt=... (API reuse path).
- **language:** `English`
- **clone:** ref_audio=yes, ref_text=yes
- **text:** Reusing a cached clone prompt should sound consistent with direct clone.
- **listen for:** Similar speaker to 07_clone_with_ref_text.

- **cuda:** ok, duration≈4.09s, peak=0.355
- **cpu:** ok, duration≈4.09s, peak=0.393

## Режимы OmniVoice (кратко)

| Mode | API inputs |
|------|------------|
| **auto** | `text` (+ optional `language`) |
| **design** | `text` + `instruct` |
| **clone** | `text` + `ref_audio` + `ref_text` (or ASR) / `voice_clone_prompt` |
| **long** | long `text` / `duration` → chunked generation |
| **control** | `speed`, `duration`, `normalize_text`, tags like `[laughter]` |

Источник API: upstream `omnivoice.models.omnivoice.OmniVoice.generate`.

