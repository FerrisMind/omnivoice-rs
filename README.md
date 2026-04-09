<!--
  README TEMPLATE — TAURI + SVELTE
  
  Замените плейсхолдеры {{...}} и удалите этот комментарий.
  
  {{PROJECT_NAME}}     — Название проекта
  {{PROJECT_SLUG}}     — GitHub slug (owner/repo)
  {{LOGO_URL}}         — URL логотипа
  {{SCREENSHOT_URL}}   — URL главного скриншота
  {{YEAR}}             — Год копирайта
  {{AUTHOR}}           — Имя автора
  {{LICENSE_TYPE}}     — Тип лицензии

  ─────────────────────────────────────────────────────────────────────────────
  ОБЯЗАТЕЛЬНЫЕ ПЕРЕВОДЫ (3 языка):
  ─────────────────────────────────────────────────────────────────────────────
  
  1. README.md      — English (основной)
  2. README.RU.md   — Русский
  3. README.PT_BR.md — Português (Brasil)
  
  Цвета бейджей:
  - Активный язык (текущий файл):
    • English:    #5B7CFA (синий)
    • Русский:    #D65C5C (красный)
    • Português:  #3ABF7A (зелёный)
  - Неактивный язык: #232323 (тёмно-серый)
-->

</p>
<p align="left">
  <!-- Для README.md (English — активный) -->
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
  
  <!-- Для README.RU.md (Русский — активный)
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
  -->
  
  <!-- Для README.PT_BR.md (Português — активный)
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
  -->
</p>

---

<p align="center">
  <img src="{{LOGO_URL}}" alt="{{PROJECT_NAME}} Logo" width="512" height="512">

<p align="center">
  <b>{{PROJECT_DESCRIPTION}}</b><br>
  {{PROJECT_TAGLINE}}
</p>

<p align="center">
  <a href="https://github.com/{{PROJECT_SLUG}}/releases"><img src="https://img.shields.io/github/v/release/{{PROJECT_SLUG}}?logo=github" alt="Latest Release"></a>
  <!-- Раскомментируйте, если проект включён в соответствующий awesome-list:
  <a href="https://github.com/tauri-apps/awesome-tauri"><img src="https://img.shields.io/badge/Awesome-Tauri-24C8D8?logo=tauri" alt="Awesome Tauri"></a>
  <a href="https://github.com/TheComputerM/awesome-svelte"><img src="https://img.shields.io/badge/Awesome-Svelte-FF3E00?logo=svelte" alt="Awesome Svelte"></a>
  <a href="https://github.com/{{PROJECT_SLUG}}/stargazers"><img src="https://img.shields.io/github/stars/{{PROJECT_SLUG}}?logo=github" alt="GitHub Stars"></a>
  -->
</p>

<h1 align="center"></h1>

<p align="center">
  <img src="{{SCREENSHOT_URL}}" alt="{{PROJECT_NAME}} Interface" width="900">
</p>

## 📚 Table of Contents

- [What is this?](#-what-is-this)
- [Demo](#-demo)
- [Key Features](#-key-features)
- [Installation & Setup](#️-installation--setup)
- [How to Start Using](#-how-to-start-using)
- [System Requirements](#️-system-requirements)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

## ✨ What is this?

{{PROJECT_NAME}} is a native desktop application for [purpose]. Built with Rust and Tauri v2, it provides a fast, [key benefit] without requiring [limitation it removes].

## 🎬 Demo

<!-- Вставьте демо-видео через GitHub assets -->
https://github.com/user-attachments/assets/your-video-id

## 🚀 Key Features

- Feature 1 — description
- Feature 2 — description
- Feature 3 — description
- Feature 4 — description
- Feature 5 — description

### Hardware Acceleration

| Backend | Status | Notes |
|---------|:------:|-------|
| CPU | ✅ | Default, works everywhere |
| CUDA (NVIDIA) | ✅ | Requires CUDA toolkit |
| Metal (Apple) | ✅ | macOS only |
| Intel MKL | ⚠️ | Optional |

## 🛠️ Installation & Setup

### Prerequisites

- Node.js (for frontend build)
- Rust toolchain (for backend)
- For CUDA: NVIDIA GPU with CUDA toolkit
- For Metal: macOS with Apple Silicon

### Development

```bash
# Install dependencies
npm install

# Run with CPU backend
npm run tauri:dev:cpu

# Run with CUDA backend (NVIDIA GPU)
npm run tauri:dev:cuda

# Platform-aware development
npm run app:dev
```

### Build

```bash
# Build with CPU backend
npm run tauri:build:cpu

# Build with CUDA backend
npm run tauri:build:cuda
```

### Quality Checks

```bash
npm run lint          # ESLint
npm run lint:fix      # ESLint with auto-fix
npm run check         # Svelte type checking
npm run format        # Prettier formatting
npm run test          # Vitest tests
```

### Rust-specific (from src-tauri/)

```bash
cargo clippy          # Linting
cargo test            # Unit tests
cargo audit           # Security audit
```

## 📖 How to Start Using

1. Build or download the application
2. Download required models/data (if applicable)
3. Launch {{PROJECT_NAME}}
4. Configure through the interface
5. Start using!

## 🖥️ System Requirements

- Windows, macOS, or Linux
- Minimum 4 GB RAM (8+ GB recommended)
- For GPU acceleration:
  - NVIDIA: CUDA-compatible GPU
  - Apple: M1/M2/M3/M4 chip (Metal)

## 🙏 Acknowledgments

This project is built on top of excellent open-source work:

- [Tauri](https://tauri.app/) — Desktop application framework
- [Svelte](https://svelte.dev/) — Frontend framework
- [Dependency](URL) — Description

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full dependency attribution.

## 📄 License

{{LICENSE_TYPE}} — see [LICENSE](LICENSE)

Copyright (c) {{YEAR}} {{AUTHOR}}
