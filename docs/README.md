# OmniVoice Rust Port Docs

This directory contains project documentation for the OmniVoice Rust port.

Unlike the upstream Python repository, this workspace is focused on inference, validation, CLI/server workflows, and runtime artifacts. Training and dataset preparation are out of scope here and should be taken from the upstream OmniVoice repository when needed.

## Document Map

- [CLI and Server](./cli-and-server.md)
  Main entrypoints, common commands, OpenAI-compatible server endpoints, and request examples.
- [Generation Parameters](./generation-parameters.md)
  Runtime generation controls implemented by the Rust port, their defaults, and practical tuning notes.
- [Voice Design](./voice-design.md)
  Supported `instruct` attributes, validation rules, and examples for design-mode synthesis.
- [Languages](./languages.md)
  How language IDs are used in this port and where to find the full language map.
- [Evaluation and Verification](./evaluation.md)
  What is actually verified in this repository, where artifacts live, and how to reproduce checks.
- [Behavior Contracts](./contracts/README.md)
  Phase-by-phase contracts for frontend, reference prompt handling, stage0, and stage1/postprocess behavior.

## Scope

This workspace documents:

- full CPU and GPU inference paths
- `omnivoice-cli` and `omnivoice-server`
- runtime model bundle layout
- parity and artifact-based verification

This workspace does not document:

- model training
- data preparation pipelines
- dataset curation
- upstream Python training/eval internals beyond what is needed for parity

## Runtime Model Layout

The checked-in local bundle is expected at `model/` and currently contains:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `audio_tokenizer/`
- `whisper/`

The local Whisper bundle under `model/whisper/` is preferred over remote fallback and currently uses the Candle-compatible GGUF layout:

- `config.json`
- `tokenizer.json`
- `whisper-large-v3-turbo-q4_0.gguf`

## Upstream Reference

For training, dataset preparation, and original model docs, see:

- [Upstream OmniVoice README](../refs/OmniVoice/README.md)
- [Upstream docs directory](../refs/OmniVoice/docs/)
