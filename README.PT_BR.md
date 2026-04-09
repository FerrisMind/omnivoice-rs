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
  <!-- Para README.md (English — ativo)
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
  -->
  
  <!-- Para README.RU.md (Русский — ativo)
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
  -->
  
  <!-- Para README.PT_BR.md (Português — ativo) -->
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
</p>

---

<p align="center">
  <img src="{{LOGO_URL}}" alt="{{PROJECT_NAME}} Logo" width="512" height="512">

<p align="center">
  <b>{{PROJECT_DESCRIPTION}}</b><br>
  {{PROJECT_TAGLINE}}
</p>

<p align="center">
  <a href="https://github.com/{{PROJECT_SLUG}}/releases"><img src="https://img.shields.io/github/v/release/{{PROJECT_SLUG}}?logo=github" alt="Última Versão"></a>
  <!-- Descomente se o projeto estiver incluído no awesome-list correspondente:
  <a href="https://github.com/tauri-apps/awesome-tauri"><img src="https://img.shields.io/badge/Awesome-Tauri-24C8D8?logo=tauri" alt="Awesome Tauri"></a>
  <a href="https://github.com/TheComputerM/awesome-svelte"><img src="https://img.shields.io/badge/Awesome-Svelte-FF3E00?logo=svelte" alt="Awesome Svelte"></a>
  <a href="https://github.com/{{PROJECT_SLUG}}/stargazers"><img src="https://img.shields.io/github/stars/{{PROJECT_SLUG}}?logo=github" alt="GitHub Stars"></a>
  -->
</p>

<h1 align="center"></h1>

<p align="center">
  <img src="{{SCREENSHOT_URL}}" alt="{{PROJECT_NAME}} Interface" width="900">
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

{{PROJECT_NAME}} é um aplicativo desktop nativo para [propósito]. Construído com Rust e Tauri v2, oferece um [benefício chave] rápido sem exigir [limitação que remove].

## 🎬 Demo

<!-- Insira o vídeo demo através dos assets do GitHub -->
https://github.com/user-attachments/assets/your-video-id

## 🚀 Principais Recursos

- Recurso 1 — descrição
- Recurso 2 — descrição
- Recurso 3 — descrição
- Recurso 4 — descrição
- Recurso 5 — descrição

### Aceleração de Hardware

| Backend | Status | Notas |
|---------|:------:|-------|
| CPU | ✅ | Padrão, funciona em todos os lugares |
| CUDA (NVIDIA) | ✅ | Requer CUDA toolkit |
| Metal (Apple) | ✅ | Apenas macOS |
| Intel MKL | ⚠️ | Opcional |

## 🛠️ Instalação e Configuração

### Pré-requisitos

- Node.js (para build do frontend)
- Rust toolchain (para backend)
- Para CUDA: GPU NVIDIA com CUDA toolkit
- Para Metal: macOS com Apple Silicon

### Desenvolvimento

```bash
# Instalar dependências
npm install

# Executar com backend CPU
npm run tauri:dev:cpu

# Executar com backend CUDA (GPU NVIDIA)
npm run tauri:dev:cuda

# Desenvolvimento com detecção de plataforma
npm run app:dev
```

### Build

```bash
# Build com backend CPU
npm run tauri:build:cpu

# Build com backend CUDA
npm run tauri:build:cuda
```

### Verificação de Qualidade

```bash
npm run lint          # ESLint
npm run lint:fix      # ESLint com auto-correção
npm run check         # Verificação de tipos Svelte
npm run format        # Formatação Prettier
npm run test          # Testes Vitest
```

### Específico para Rust (de src-tauri/)

```bash
cargo clippy          # Linting
cargo test            # Testes unitários
cargo audit           # Auditoria de segurança
```

## 📖 Como Começar a Usar

1. Compile ou baixe o aplicativo
2. Baixe modelos/dados necessários (se aplicável)
3. Inicie {{PROJECT_NAME}}
4. Configure através da interface
5. Comece a usar!

## 🖥️ Requisitos do Sistema

- Windows, macOS ou Linux
- Mínimo 4 GB de RAM (8+ GB recomendado)
- Para aceleração GPU:
  - NVIDIA: GPU compatível com CUDA
  - Apple: chip M1/M2/M3/M4 (Metal)

## 🙏 Agradecimentos

Este projeto é construído sobre excelente trabalho de código aberto:

- [Tauri](https://tauri.app/) — Framework de aplicativo desktop
- [Svelte](https://svelte.dev/) — Framework frontend
- [Dependency](URL) — Descrição

Veja [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) para atribuição completa de dependências.

## 📄 Licença

{{LICENSE_TYPE}} — veja [LICENSE](LICENSE)

Copyright (c) {{YEAR}} {{AUTHOR}}
