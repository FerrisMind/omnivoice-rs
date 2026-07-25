<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>OmniVoice Rust Port</b><br>
  GPU-first Rust workspace для инференса OmniVoice, проверки паритета и CLI-запуска на Candle.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-workspace-DEA584?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Candle-0.10.2-232323" alt="Candle">
  <img src="https://img.shields.io/badge/Inference-GPU--First-5B7CFA" alt="GPU First">
</p>

## 📚 Содержание

- [Что это?](#-что-это)
- [Демо](#-демо)
- [Прослушка](#-прослушка)
- [Ключевые возможности](#-ключевые-возможности)
- [Установка и настройка](#️-установка-и-настройка)
- [Как начать использовать](#-как-начать-использовать)
- [Системные требования](#️-системные-требования)
- [Благодарности](#-благодарности)
- [Лицензия](#-лицензия)

## ✨ Что это?

Это Rust workspace для портирования инференса OmniVoice на Candle с приоритетом GPU-исполнения.

Состав:

- `crates/omnivoice-infer` — двухстейджевый inference pipeline
- `crates/omnivoice-cli` — CLI для подготовки промпта, инспекции stage0/stage1 и полного инференса
- `docs/contracts` — поведенческие контракты по фазам
- `tools` — локальные вспомогательные скрипты

CUDA и Metal остаются предпочтительными backend’ами ускорения, но полноценный OmniVoice inference доступен как на CPU, так и на GPU.

## 🎬 Демо

Публичного GUI-демо в этом workspace нет. Основной интерфейс — CLI.

```powershell
cargo run -p omnivoice-cli --features cuda -- infer `
  --text "Hello, this is a test of zero-shot text-to-speech." `
  --language en `
  --output artifacts\demo.wav `
  --device cuda:0 `
  --dtype f32 `
  --seed 1234
```

Если нужно явно указать локальный bundle или конкретный Hugging Face repo, передайте `--model`:

```powershell
cargo run -p omnivoice-cli --features cuda -- infer `
  --model model `
  --text "Hello from a local model bundle." `
  --output artifacts\demo-local.wav
```

## 🎧 Прослушка

Готовые WAV для сравнения **эталонного Python OmniVoice** и этого **Rust**-порта на одних и тех же промптах и seed.

| | Путь |
|--|------|
| Сводная таблица A/B | [`artifacts/LISTENING_COMPARE.md`](artifacts/LISTENING_COMPARE.md) |
| Python-эталон | [`artifacts/listening_demo/`](artifacts/listening_demo/) |
| Rust-порт | [`artifacts/listening_demo_rust/`](artifacts/listening_demo_rust/) |

Быстрые пары CUDA:

| Сценарий | Python | Rust |
|----------|--------|------|
| Auto EN | [wav](artifacts/listening_demo/cuda/01_auto_en_short.wav) | [wav](artifacts/listening_demo_rust/cuda_0/01_auto_en_short.wav) |
| Design British ♀ | [wav](artifacts/listening_demo/cuda/04_design_en_british_female.wav) | [wav](artifacts/listening_demo_rust/cuda_0/04_design_en_british_female.wav) |
| Clone + ref text | [wav](artifacts/listening_demo/cuda/07_clone_with_ref_text.wav) | [wav](artifacts/listening_demo_rust/cuda_0/07_clone_with_ref_text.wav) |
| Chinese + laughter | [wav](artifacts/listening_demo/cuda/06_design_zh_nonverbal.wav) | [wav](artifacts/listening_demo_rust/cuda_0/06_design_zh_nonverbal.wav) |

Полная таблица из 15 сценариев — в [LISTENING_COMPARE.md](artifacts/LISTENING_COMPARE.md).

## 🚀 Ключевые возможности

- Двухстейджевый OmniVoice inference pipeline на Rust
- GPU-first выбор runtime: CUDA -> Metal -> CPU fallback
- Полноценные inference-пути доступны и на CPU, и на GPU
- Stage0 и Stage1 parity harness против официальных reference-артефактов OmniVoice
- Voice clone, voice design, auto voice, batch и long-form chunked inference
- CLI-команды `prepare-prompt`, `stage0-debug`, `stage1-decode` и `infer`
- Отдельный `omnivoice-server` с `/`, `/health`, `/v1/models` и `/v1/audio/speech`
- Серверный readiness-статус, монтирование под base path, CORS/OPTIONS и поддержка JSON или multipart для TTS-запросов

### Аппаратное ускорение

| Backend | Статус | Примечания |
|---------|:------:|------------|
| CPU | ✅ | Полноценный inference подтверждён локально; подходит для CPU-only запуска и отладки |
| CUDA (NVIDIA) | ✅ | Полноценный inference подтверждён; основной путь GPU-ускорения |
| Metal (Apple) | ✅ | GPU backend для macOS; реализован и отражён в тестах |

## 🛠️ Установка и настройка

### Требования

- Rust toolchain
- CPU-only inference поддерживается; GPU-ускорение опционально
- Для CUDA: NVIDIA GPU и совместимый драйвер/toolkit
- Для Metal: macOS с поддержкой Metal
- Либо локальный OmniVoice bundle, либо network access для скачивания моделей в Hugging Face cache
- Локально доступные официальные upstream-референсы

### Разработка

```powershell
cargo fmt
cargo test -p omnivoice-infer --features cuda --test phase_status -- --nocapture --test-threads=1
cargo test -p omnivoice-cli --features cuda --test phase10_cli_cuda -- --nocapture --test-threads=1
```

### Проверка качества

```powershell
cargo fmt
cargo clippy --workspace --all-targets
cargo test -p omnivoice-infer --features cuda --test phase10_cuda_acceptance -- --nocapture --test-threads=1
cargo test -p omnivoice-cli --features cuda --test phase10_cli_cuda -- --nocapture --test-threads=1
```

## 📖 Как начать использовать

1. Либо передайте `--model <local-path>`, либо дайте CLI/server автоматически разрешить `k2-fsa/OmniVoice` из Hugging Face.
2. Если не передавать `--asr-model`, Whisper по умолчанию берётся из `oxide-lab/whisper-large-v3-turbo-GGUF` и скачивает Candle-совместимый набор `config.json`, `tokenizer.json`, `whisper-large-v3-turbo-q4_0.gguf`.
3. Держите официальные upstream-референсы локально доступными для проверки поведения.
4. Запускайте GPU-тесты последовательно, а не все сразу.
5. Используйте `omnivoice-cli infer` для end-to-end синтеза.

## 🖥️ Системные требования

- Windows, macOS или Linux
- Достаточно RAM и VRAM для весов OmniVoice и runtime tensors
- Для GPU-ускорения:
  - NVIDIA GPU для CUDA
  - Apple Silicon / поддерживаемый macOS GPU для Metal

## 🙏 Благодарности

Этот workspace опирается на официальные upstream-проекты:

- [OmniVoice](https://github.com/k2-fsa/OmniVoice)
- [candle](https://github.com/huggingface/candle)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)

Официальные upstream-проекты остаются источником правды для поведения и инженерного reference, но как часть этого репозитория не отслеживаются.

См. [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) для целевой атрибуции основных upstream-проектов, релевантных этому workspace.

## 📄 Лицензия

Apache-2.0 — см. [LICENSE](LICENSE)

Copyright (c) 2026 FerrisMind
