# Third-Party Licenses

This workspace depends on and/or adapts behavior from several upstream open-source projects.

This file is not a full machine-generated dependency inventory. It is a focused attribution file for the primary upstream projects that are materially relevant to this repository's source code, inference behavior, or distributed functionality.

## Primary Upstream Projects

| Project | Role in this repository | Upstream | License |
|---------|-------------------------|----------|---------|
| OmniVoice | Behavioral source of truth for inference, prompt preparation, chunking, and postprocess logic | https://github.com/k2-fsa/OmniVoice | Apache-2.0 |
| candle | Rust ML runtime used by this workspace; local workspace also references Whisper mel-filter assets from Candle examples | https://github.com/huggingface/candle | Apache-2.0 or MIT |
| mistral.rs | Crates.io dependencies from the mistral.rs ecosystem are used for audio/quantization support (`mistralrs-quant`, optional `mistralrs-audio`) | https://github.com/EricLBuehler/mistral.rs | MIT |
| shine-rs | MP3 encoding for `omnivoice-server` OpenAI-compatible speech responses | https://github.com/wshon/shine-rs | LGPL-2.0 |

## Notes

- Local upstream checkouts are used as engineering reference material and are not part of version control.
- This repository does not vendor the full source trees of `OmniVoice`, `candle`, or `mistral.rs`.
- When code or behavior is adapted from upstream, the corresponding upstream license remains applicable to those adapted portions.
- For a complete dependency-level inventory of the final binary distribution, generate a crate-level report separately from the exact locked dependency graph used for release builds.

## OmniVoice

Upstream project: `k2-fsa/OmniVoice`

License: Apache License 2.0

Relevant use in this repository:

- inference behavior and parity target
- prompt preparation semantics
- chunked generation semantics
- postprocess behavior

License reference:

- upstream project metadata declares `Apache-2.0`
- local development should retain a copy of the upstream license text when OmniVoice-derived code is adapted

## candle

Upstream project: `huggingface/candle`

License: dual-licensed under Apache-2.0 or MIT

Relevant use in this repository:

- `candle-core`, `candle-nn`, `candle-transformers`
- local Whisper mel-filter assets referenced from Candle example data

License references:

- Candle is dual-licensed under Apache-2.0 or MIT upstream
- local development should retain copies of the upstream license texts when Candle-derived assets or code are redistributed

MIT notice for Candle:

```text
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## mistral.rs

Upstream project: `EricLBuehler/mistral.rs`

License: MIT

Relevant use in this repository:

- crates.io dependency `mistralrs-quant`
- optional crates.io dependency `mistralrs-audio`
- local upstream source used as implementation reference only

MIT notice for mistral.rs:

```text
MIT License

Copyright (c) 2024 Eric Buehler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## shine-rs

Upstream project: `wshon/shine-rs`

License: LGPL-2.0

Relevant use in this repository:

- `omnivoice-server` MP3 encoding for `/v1/audio/speech`

License reference:

- upstream crate metadata declares `LGPL-2.0`
- downstream binary distribution should preserve the corresponding license notice and compliance obligations for the encoder dependency
