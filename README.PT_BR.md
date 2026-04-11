<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
</p>

---

<p align="center">
  <b>OmniVoice Rust Port</b><br>
  Workspace Rust com foco em GPU para inferência OmniVoice, validação de paridade e execução via CLI com Candle.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-workspace-DEA584?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Candle-0.10.2-232323" alt="Candle">
  <img src="https://img.shields.io/badge/Inference-GPU--First-5B7CFA" alt="GPU First">
</p>

## 📚 Índice

- [O que é isso?](#-o-que-é-isso)
- [Demo](#-demo)
- [Principais Recursos](#-principais-recursos)
- [Instalação e Configuração](#️-instalação-e-configuração)
- [Como Começar a Usar](#-como-começar-a-usar)
- [Requisitos do Sistema](#️-requisitos-do-sistema)
- [Agradecimentos](#-agradecimentos)
- [Licença](#-licença)

## ✨ O que é isso?

Este é um workspace Rust para portar a inferência do OmniVoice para Candle com execução prioritária em GPU.

Ele contém:

- `crates/omnivoice-infer` — pipeline de inferência em dois estágios
- `crates/omnivoice-cli` — CLI para preparação de prompt, inspeção de stage0/stage1 e inferência completa
- `docs/contracts` — contratos de comportamento por fase
- `tools` — scripts locais de apoio

CUDA e Metal são os backends principais. CPU existe apenas para fallback, offload explícito, preprocessing e materialização de debug.

## 🎬 Demo

Não há demo GUI pública neste workspace. A interface principal é a CLI.

```powershell
cargo run -p omnivoice-cli --features cuda -- infer `
  --model-dir H:\omnivoice\model `
  --text "Hello, this is a test of zero-shot text-to-speech." `
  --language en `
  --output H:\omnivoice\artifacts\demo.wav `
  --device cuda:0 `
  --dtype f32 `
  --seed 1234
```

## 🚀 Principais Recursos

- Pipeline OmniVoice em dois estágios escrito em Rust
- Seleção de runtime GPU-first: CUDA -> Metal -> CPU fallback
- Harnesses de paridade Stage0 e Stage1 contra referências oficiais do OmniVoice
- Voice clone, voice design, auto voice, batch e inferência chunked para textos longos
- Fluxos CLI para `prepare-prompt`, `stage0-debug`, `stage1-decode` e `infer`

### Aceleração de Hardware

| Backend | Status | Notas |
|---------|:------:|-------|
| CPU | ✅ | Fallback, preprocessing, debug, offload |
| CUDA (NVIDIA) | ✅ | Caminho principal de validação local |
| Metal (Apple) | ✅ | Implementado e espelhado nos testes |

## 🛠️ Instalação e Configuração

### Pré-requisitos

- Rust toolchain
- Para CUDA: GPU NVIDIA e driver/toolkit compatível
- Para Metal: macOS com suporte a Metal
- Pesos locais do modelo em `model/`
- Materiais oficiais de referência upstream disponíveis localmente

### Desenvolvimento

```powershell
cargo fmt
cargo test -p omnivoice-infer --features cuda --test phase_status -- --nocapture --test-threads=1
cargo test -p omnivoice-cli --features cuda --test phase10_cli_cuda -- --nocapture --test-threads=1
```

### Verificação de Qualidade

```powershell
cargo fmt
cargo clippy --workspace --all-targets
cargo test -p omnivoice-infer --features cuda --test phase10_cuda_acceptance -- --nocapture --test-threads=1
cargo test -p omnivoice-cli --features cuda --test phase10_cli_cuda -- --nocapture --test-threads=1
```

## 📖 Como Começar a Usar

1. Coloque os assets reais do modelo em `model/`.
2. Se você não passar `--asr-model`, o Whisper usará por padrão `oxide-lab/whisper-large-v3-turbo-GGUF` e baixará o conjunto compatível com Candle: `config.json`, `tokenizer.json` e `whisper-large-v3-turbo-q4_0.gguf`.
3. Mantenha os materiais oficiais de referência upstream disponíveis localmente para verificação de comportamento.
4. Execute os testes de GPU de forma sequencial, não todos ao mesmo tempo.
5. Use `omnivoice-cli infer` para síntese end-to-end.

## 🖥️ Requisitos do Sistema

- Windows, macOS ou Linux
- RAM e VRAM suficientes para os pesos OmniVoice e os tensores de runtime
- Para aceleração GPU:
  - GPU NVIDIA para CUDA
  - Apple Silicon / GPU compatível no macOS para Metal

## 🙏 Agradecimentos

Este workspace é construído sobre projetos upstream oficiais:

- [OmniVoice](https://github.com/k2-fsa/OmniVoice)
- [candle](https://github.com/huggingface/candle)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)

Os projetos upstream oficiais continuam sendo a fonte de verdade para comportamento e referência de engenharia, mas não são versionados como parte deste repositório.

Veja [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) para a atribuição focada dos principais projetos upstream relevantes para este workspace.

## 📄 Licença

Apache-2.0 — veja [LICENSE](LICENSE)

Copyright (c) 2026 FerrisMind
