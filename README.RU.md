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

CUDA и Metal считаются основными backend’ами. CPU используется только как fallback, explicit offload, preprocessing и debug materialization.

## 🎬 Демо

Публичного GUI-демо в этом workspace нет. Основной интерфейс — CLI.

```powershell
cargo run -p omnivoice-cli --features cuda -- infer `
  --text "Hello, this is a test of zero-shot text-to-speech." `
  --language en `
  --output H:\omnivoice\artifacts\demo.wav `
  --device cuda:0 `
  --dtype f32 `
  --seed 1234
```

Если нужно явно указать локальный bundle или конкретный Hugging Face repo, передайте `--model`:

```powershell
cargo run -p omnivoice-cli --features cuda -- infer `
  --model H:\models\OmniVoice `
  --text "Hello from a local model bundle." `
  --output H:\omnivoice\artifacts\demo-local.wav
```

## 🚀 Ключевые возможности

- Двухстейджевый OmniVoice inference pipeline на Rust
- GPU-first выбор runtime: CUDA -> Metal -> CPU fallback
- Stage0 и Stage1 parity harness против официальных reference-артефактов OmniVoice
- Voice clone, voice design, auto voice, batch и long-form chunked inference
- CLI-команды `prepare-prompt`, `stage0-debug`, `stage1-decode` и `infer`

### Аппаратное ускорение

| Backend | Статус | Примечания |
|---------|:------:|------------|
| CPU | ✅ | Fallback, preprocessing, debug, offload |
| CUDA (NVIDIA) | ✅ | Основной локальный путь валидации |
| Metal (Apple) | ✅ | Реализован и отражён в тестах |

## 🛠️ Установка и настройка

### Требования

- Rust toolchain
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
