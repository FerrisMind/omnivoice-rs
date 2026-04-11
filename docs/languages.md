# Languages

The OmniVoice Rust port uses the same language ID map as the upstream OmniVoice model family.

The checked-in language map lives in:

- [`docs/lang_id_name_map.tsv`](./lang_id_name_map.tsv)

This TSV includes:

- `language_id`
- `language_name`
- `iso_639_3_id`
- `train_data_duration`

## How Language Selection Works

For CLI inference, pass a language ID or known short form through `--language`.

Example:

```powershell
cargo run -p omnivoice-cli -- infer `
  --model model `
  --text "Hello world" `
  --language en `
  --output out\hello.wav `
  --device cpu `
  --dtype f32
```

In server requests, use the `language` extension field.

## Examples

Common entries from the map:

| Language | OmniVoice ID | ISO 639-3 |
|---|---|---|
| English | `en` | `eng` |
| Chinese | `zh` | `cmn` |
| Cantonese | `yue` | `yue` |
| Portuguese | `pt` | `por` |
| Russian | `ru` | `rus` |

## Notes

- the Rust port uses the same language IDs for prompt construction and verification artifacts
- language choice affects prompt formatting and some voice-design normalization rules
- English accents and Chinese dialects are only meaningful for their corresponding language families

## Full Table

Use the TSV directly for the complete list. It is intentionally kept as a machine-readable source of truth rather than duplicated into a giant Markdown table.
